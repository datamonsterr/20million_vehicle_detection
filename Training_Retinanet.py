import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch
import torchvision

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

folder = 'nighttime'
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

# Read the dataset
imgs_data = []
for img_path in tqdm(img_paths):
    img = cv2.imread(img_path)
    imgs_data.append([img_path, img])

# Read the labels
labels_data = []
for anno_path in tqdm(annotation_paths):
    with open(anno_path, 'r') as file:
        lines = file.readlines()

    labels = []
    for line in lines:
        parts = line.strip().split()  # Split the line by spaces
        class_id = int(parts[0]) - 3  # Convert class_id to integer
        x_center = float(parts[1])    # Convert x_center to float
        y_center = float(parts[2])    # Convert y_center to float
        width = float(parts[3])       # Convert width to float
        height = float(parts[4])      # Convert height to float
        labels.append([class_id, x_center, y_center, width, height])

    labels_data.append([anno_path, labels])

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

def visualize_gt_img(img, ground_truths):
    height, width, _ = img.shape

    for gt in ground_truths:
        class_id, x_center, y_center, w, h = gt
        x1, y1, x2, y2 = convert_to_pixel_coords(x_center, y_center, w, h, width, height)
        # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        # Add class label
        cv2.putText(img, f"Class {class_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image
    plt.imshow(img)
    plt.axis('off')
    plt.show

def visualize_pred_img(img, pred, threshold=0.8):
    height, width, _ = img.shape

    keys = list(pred.keys())
    values = list(pred.values())

    bboxes = values[0].detach().numpy()
    scores = values[1].detach().numpy()
    labels = values[2].detach().numpy()

    for i in range(len(bboxes)):
        if scores[i] >= threshold:
            x_min, y_min, x_max, y_max = bboxes[i]
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
            cv2.putText(img, f"Class {labels[i]}", (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    plt.imshow(img)
    plt.axis('off')
    plt.show()

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((720, 1280)),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def yolo_to_normal(x_center, y_center, width, height, img_w, img_h):
    xmin = (x_center - width / 2) * img_w
    ymin = (y_center - height / 2) * img_h
    xmax = (x_center + width / 2) * img_w
    ymax = (y_center + height / 2) * img_h
    return xmin, ymin, xmax, ymax

class TrafficDataset(Dataset):
    def __init__(self, images, annotations, trans=None):
        self.images = images
        self.annotations = annotations
        self.trans = trans

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        bboxes = []
        labels = []

        image = self.images[idx][1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_heigth, img_width, _ = image.shape

        # Get labels and bboxes
        annotation = self.annotations[idx][1]
        for anno in annotation:
            labels.append(anno[0])
            x_center, y_center, w, h = anno[1:]
            x_min, y_min, x_max, y_max = yolo_to_normal(x_center, y_center, w, h, img_width, img_heigth)
            bboxes.append([x_min, y_min, x_max, y_max])

        labels = torch.tensor(labels, dtype=torch.int64)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)

        targets = {
            'boxes': bboxes,
            'labels': labels
        }

        if self.trans == None:
            image = trans(image)

        return image, targets
    
X_train, X_test, Y_train, Y_test = train_test_split(imgs_data, labels_data, test_size=0.2, shuffle=True, random_state=42)

def collate_fn(batch):
    return tuple(zip(*batch))

dataset = TrafficDataset(X_train, Y_train)
train_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

model = torchvision.models.detection.retinanet_resnet50_fpn(weights='COCO_V1', progress=True)

# Modify the number of classes (including background)
num_classes = 5         # Include background class
in_features = list(model.head.classification_head.conv)[0][0].in_channels
num_anchors = model.head.classification_head.num_anchors
print(num_anchors)

model.head.classification_head = RetinaNetClassificationHead(
    in_channels=in_features,
    num_anchors=num_anchors,
    num_classes=num_classes
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lr = 1e-4
num_epochs = 20

def trainingloop(model, train_loader, num_epochs):
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

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

trainingloop(model=model, train_loader=train_loader, num_epochs=num_epochs)