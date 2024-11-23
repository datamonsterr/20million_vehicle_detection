import os
import argparse
import cv2
import torch
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Function to convert normalized coordinates to pixel values
def convert_to_pixel_coords(x_center, y_center, width, height, img_width, img_height):
    x_center_pixel = int(x_center * img_width)
    y_center_pixel = int(y_center * img_height)
    width_pixel = int(width * img_width)
    height_pixel = int(height * img_height)

    # Calculate the top-left and bottom-right corners of the bounding box
    x1 = x_center_pixel - width_pixel // 2
    y1 = y_center_pixel - height_pixel // 2
    x2 = x_center_pixel + width_pixel // 2
    y2 = y_center_pixel + height_pixel // 2

    return x1, y1, x2, y2


def yolo_to_normal(x_center, y_center, width, height, img_w, img_h):
    xmin = (x_center - width / 2) * img_w
    ymin = (y_center - height / 2) * img_h
    xmax = (x_center + width / 2) * img_w
    ymax = (y_center + height / 2) * img_h
    return xmin, ymin, xmax, ymax

class TrafficDataset(Dataset):
    def __init__(self, image_paths, annotations, trans=None):
        self.image_paths = image_paths 
        self.annotations = annotations
        self.trans = trans

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_heigth, img_width, _ = image.shape

        bboxes = []
        labels = []

        # Get labels and bboxes
        annotation = self.annotations[idx][1]
        for anno in annotation:
            label, x_center, y_center, w, h = anno
            labels.append(label)
            x_min, y_min, x_max, y_max = yolo_to_normal(x_center, y_center, w, h, img_width, img_heigth)
            bboxes.append([x_min, y_min, x_max, y_max])

        labels = torch.tensor(labels, dtype=torch.int64)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)

        targets = {
            'boxes': bboxes,
            'labels': labels
        }

        if self.trans:
            image = self.trans(image)

        return image, targets
    

def collate_fn(batch):
    return tuple(zip(*batch))

def trainingloop(model, train_loader, num_epochs, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    model.train()
    loss_list = []
    for epoch in range(num_epochs):
        losses = 0.0
        for i, (images, targets) in tqdm(enumerate(train_loader)):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item():.4f}")
        torch.save(model.state_dict(), './checkpoint/Retinanet_nighttime.pt')
        loss_list.append(losses)


def main():
    folder = './dataset/original/train/nighttime'
    img_paths = []
    annotation_paths = []

    current_directory = os.getcwd()
    nighttime_path = os.path.join(current_directory, folder)

    for img_name in sorted(os.listdir(nighttime_path))[:7000]:
        if img_name.endswith('.jpg'):
            img_path = os.path.join(nighttime_path, img_name)
            img_paths.append(img_path)
        elif img_name.endswith('.txt'):
            annotation_path = os.path.join(nighttime_path, img_name)
            annotation_paths.append(annotation_path)

    labels_data = []
    for anno_path in tqdm(annotation_paths):
        with open(anno_path, 'r') as file:
            lines = file.readlines()

        labels = []
        for line in lines:
            parts = line.strip().split()  # Split the line by spaces
            class_id = int(parts[0])  # Convert class_id to integer
            x_center = float(parts[1])    # Convert x_center to float
            y_center = float(parts[2])    # Convert y_center to float
            width = float(parts[3])       # Convert width to float
            height = float(parts[4])      # Convert height to float
            labels.append([class_id, x_center, y_center, width, height])
        labels_data.append([anno_path, labels])

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((720, 1280)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    parser = argparse.ArgumentParser(description="RetinaNet model object detection script")
    parser.add_argument('--batch_size', type=int, default=8, help="Set batch size")
    parser.add_argument('--num_workers', type=int, default=4, help="Set number of workers")
    parser.add_argument('--epoch', type=int, default=100, help="Set number of workers")
    parser.add_argument('--lr', type=float, default=0.01, help="Set number of workers")
    args = parser.parse_args()

    X_train = img_paths
    Y_train = labels_data
    dataset = TrafficDataset(X_train, Y_train, trans=trans)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=collate_fn)
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights='COCO_V1', progress=True)

    # Modify the number of classes (including background)
    num_classes = 5         # Include background class
    in_features = list(model.head.classification_head.conv)[0][0].in_channels
    num_anchors = model.head.classification_head.num_anchors

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_features,
        num_anchors=num_anchors,
        num_classes=num_classes
    )


    trainingloop(model=model, train_loader=train_loader, num_epochs=args.epoch, lr=args.lr)



if __name__ == "__main__":
    main()