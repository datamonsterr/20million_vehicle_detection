import argparse
import os
import random
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from tqdm import tqdm


def yolo_to_normal(x_center, y_center, width, height, img_w, img_h):
    xmin = (x_center - width / 2) * img_w
    ymin = (y_center - height / 2) * img_h
    xmax = (x_center + width / 2) * img_w
    ymax = (y_center + height / 2) * img_h
    return xmin, ymin, xmax, ymax

class VehiclesDetectionDataset(Dataset):
    def __init__(self, image_paths, annotations, img_width, img_height, transforms=None):
        self.image_paths = image_paths
        self.annotations = annotations
        self.transforms = transforms
        self.img_width = img_width
        self.img_height = img_height

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = torchvision.io.read_image(img_path)
        img = img.float() / 255.0

        # Get annotation for this image
        ann = self.annotations[idx]
        boxes = []
        labels = []

        for bbox in ann:
            class_id, x_center, y_center, width, height = bbox
            class_id += 1
            # Convert normalized coordinates to pixel coordinates
            xmin, ymin, xmax, ymax = yolo_to_normal(x_center, y_center, width, height, self.img_width, self.img_height)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_id)
        # Convert boxes and labels to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Create the target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
        }
        # Apply any transformations to the image
        if self.transforms:
            img = self.transforms(img)
        return img, target

def read_annotation(annotation_path):
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
        ann = []
        for line in lines:
            line = line.strip().split(' ')
            class_id = int(line[0])
            x_center = float(line[1])
            y_center = float(line[2])
            width = float(line[3])
            height = float(line[4])

            # Relabel nighttime class -> daytime
            ann.append([class_id, x_center, y_center, width, height])
        return ann

def load_train_data(train_path):
    annotations = {'day': [], 'night': []}
    img_paths = {'day': [], 'night': []}
    for t in ['day', 'night']:
        for root, _, files in os.walk(f"{train_path}/{t}time"):
            for file in files:
                if file.endswith(".jpg"):
                    if os.path.exists(os.path.join(root, file.replace(".jpg", ".txt"))):
                        img_paths[t].append(os.path.join(root,file))
                        annotations[t].append(read_annotation(os.path.join(root, file.replace(".jpg", ".txt"))))
    return img_paths, annotations
        
def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    devicetype = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.GradScaler(devicetype)
    model.train()
    train_loss_list = []
    tqdm_bar = tqdm(data_loader, total=len(data_loader))

    for idx, data in enumerate(tqdm_bar):
        optimizer.zero_grad()
        images, targets = data
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.autocast(devicetype):  # Mixed precision
            losses = model(images, targets)
            loss = sum(loss for loss in losses.values())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if idx % 10 == 0:  # update every 10 batches
            tqdm_bar.set_description(desc=f"Training Loss: {loss:.3f}")

        train_loss_list.append(loss.item())
    return train_loss_list

def train(model, optimizer, lr_scheduler, data_loader, num_epochs, output_dir, filename):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    loss_dict = {'train_loss': [], 'valid_loss': []}

    for epoch in range(0, num_epochs):
      print("----------Epoch {}----------".format(epoch+1))
      train_loss_list = train_one_epoch(model, optimizer, data_loader, device, epoch)
      loss_dict['train_loss'].extend(train_loss_list)
      lr_scheduler.step()
      ckpt_file_name = f"{output_dir}/{filename}_e{epoch+1}.pth"
      torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_dict': loss_dict
      }, ckpt_file_name)

    print("Training Finished !")

def main():
    SEED = 17 
    classes = ['motorbike', 'car','bus', 'container']
    NUM_CLASSES = len(classes) + 1 

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    img_paths, annotations = load_train_data('dataset/original/train')
    img_width = 1280
    img_height = 720
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all", help="all | day | night")
    parser.add_argument("--v2", type=bool, default=False, help="fasterrcnn v1 or v2")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for the data loader")
    parser.add_argument("--output_dir", type=str, default="./checkpoint", help="Output directory")
    parser.add_argument("--latest_checkpoint", type=str, default=None, help="Path to the latest checkpoint")
    parser.add_argument("--epoch", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--lr_momentum", type=float, default=0.9, help="Learning rate momentum")
    parser.add_argument("--lr_decay_rate", type=float, default=0.001, help="Learning rate decay rate")
    parser.add_argument("--lr_sched_step_size", type=int, default=1, help="Learning rate scheduler step size")
    parser.add_argument("--lr_sched_gamma", type=float, default=0.1, help="Learning rate scheduler gamma")
    
    args = parser.parse_args()

    LR = args.lr
    LR_MOMENTUM = args.lr_momentum
    LR_DECAY_RATE = args.lr_decay_rate
    LR_SCHED_STEP_SIZE = args.lr_sched_step_size
    LR_SCHED_GAMMA = args.lr_sched_gamma
    NUM_EPOCHS = args.epoch

    args = parser.parse_args()
    dataset_night = VehiclesDetectionDataset(img_paths['night'], annotations['night'], img_width, img_height)
    dataset_day = VehiclesDetectionDataset(img_paths['day'], annotations['day'], img_width, img_height)
    dataset = None
    if args.dataset == "day":
        dataset = dataset_day
    elif args.dataset == "night":
        dataset = dataset_night
    else:
        dataset = ConcatDataset([dataset_day, dataset_night])
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=True, collate_fn=collate_fn)


    if args.v2:
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    else:
        model = fasterrcnn_resnet50_fpn(FasterRCNN_ResNet50_FPN_Weights.DEFAULT)


    model.backbone.zero_grad(True)
    for name, param in model.backbone.body.named_parameters():
        if "layer4" not in name:
          param.requires_grad = False
        else:
          param.requires_grad = True

    for param in model.rpn.parameters():
        param.requires_grad = True

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=LR, momentum=LR_MOMENTUM, weight_decay=LR_DECAY_RATE)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_SCHED_STEP_SIZE,gamma=LR_SCHED_GAMMA)
    
    train(model, optimizer, lr_scheduler, data_loader, NUM_EPOCHS, output_dir=args.output_dir, filename=f"faster_rcnn_{args.dataset}_ep{NUM_EPOCHS}_lr{LR}")
    
if __name__ == "__main__":
    main()