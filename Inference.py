import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse

import torch
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
import supervision as sv
from torchvision.ops import nms
from torchvision.utils import draw_bounding_boxes
from ensemble_boxes import weighted_boxes_fusion

from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

folder_path = os.getcwd()
print(folder_path)
sys.path.append(folder_path)

folder_name = 'public_test'

img_paths = []
for img_name in os.listdir(os.path.join(folder_path, folder_name)):
    img_paths.append(os.path.join(folder_path, folder_name, img_name))

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

def visualize_pred_img(img, pred, threshold=0.45):
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
            if labels[i] == 0:
                cv2.putText(img, f"Class {labels[i]} - {scores[i]:.2f}", (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
            if labels[i] == 1:
                cv2.putText(img, f"Class {labels[i]} - {scores[i]:.2f}", (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            if labels[i] == 2:
                cv2.putText(img, f"Class {labels[i]} - {scores[i]:.2f}", (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 240, 172), 2)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (100, 240, 172), 2)
            if labels[i] == 3:
                cv2.putText(img, f"Class {labels[i]} - {scores[i]:.2f}", (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 120, 88), 2)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (200, 120, 88), 2)

    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def visualize_gt_test(img, ground_truth):
    height, width, _ = img.shape

    keys = list(ground_truth.keys())
    values = list(ground_truth.values())

    bboxes = values[0].detach().numpy()
    labels = values[1].detach().numpy()

    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = bbox
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img, f"Class {labels[i]}", (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((720, 1280))
    ])

class TrafficDataset(Dataset):
    def __init__(self, images, trans=None):
        self.images = images
        self.trans = trans

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_heigth, img_width, _ = image.shape

        if self.trans == None:
            image = trans(image)

        return image


class TrafficPathDataset(Dataset):
    def __init__(self, image_paths, trans=None):
        self.image_paths = image_paths
        self.trans = trans

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.trans == None:
            image = trans(image)

        return image

filenames_day = []
filenames_night = []

for img_path in img_paths:
    if 'cam_10' in img_path or 'src_1_frame' in img_path:
        filenames_day.append(img_path)
    else:
        filenames_night.append(img_path)
    
imgs_day = []
imgs_night = []

for filepath in tqdm(filenames_day):
    img = cv2.imread(filepath)
    imgs_day.append(img)

for filepath in tqdm(filenames_night):
    img = cv2.imread(filepath)
    imgs_night.append(img)
    
img_dark_night_paths = []
img_night_paths = []
for filename_path in sorted(filenames_night):
    if 'src_2_frame' in os.path.basename(filename_path):
        img_dark_night_path = os.path.join(folder_path, 'enhanced_images', os.path.basename(filename_path))
        img_dark_night_paths.append(img_dark_night_path)
    else:
        img_night_path = os.path.join(folder_path, 'images', os.path.basename(filename_path))
        img_night_paths.append(img_night_path)

img_bright_night_paths = img_dark_night_paths[261:]
img_dark_night_paths = img_dark_night_paths[:261]

imgs_dark_night = []
imgs_bright_night = []
imgs_night = []

for img_dark_night_path in tqdm(img_dark_night_paths):
    img = cv2.imread(img_dark_night_path)
    imgs_dark_night.append(img)

for img_bright_night in tqdm(img_bright_night_paths):
    img = cv2.imread(img_bright_night)
    imgs_bright_night.append(img)

for img_night_path in tqdm(img_night_paths):
    img = cv2.imread(img_night_path)
    imgs_night.append(img)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RetinaNet
retinanet_nighttime = torchvision.models.detection.retinanet_resnet50_fpn(weights='COCO_V1', progress=True)
# Modify the number of classes (including background)
num_classes = 5         # Include background class
in_features = list(retinanet_nighttime.head.classification_head.conv)[0][0].in_channels
num_anchors = retinanet_nighttime.head.classification_head.num_anchors

retinanet_nighttime.head.classification_head = RetinaNetClassificationHead(
    in_channels=in_features,
    num_anchors=num_anchors,
    num_classes=num_classes
)
retinanet_nighttime.load_state_dict(torch.load('./checkpoint/Retinanet_nighttime.pt', map_location='cpu'))

# Faster R-CNN Daytime
faster_rcnn_daytime = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
in_features = faster_rcnn_daytime.roi_heads.box_predictor.cls_score.in_features
faster_rcnn_daytime.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 5)
checkpoint = torch.load('./checkpoint/epoch_5_model.pth', map_location='cpu')
faster_rcnn_daytime.load_state_dict(checkpoint['model_state_dict'])

# Faster R-CNN Nighttime
faster_rcnn_nighttime = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
in_features = faster_rcnn_nighttime.roi_heads.box_predictor.cls_score.in_features
faster_rcnn_nighttime.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 5)
checkpoint = torch.load('./checkpoint/faster_rcnn_night.pth', map_location='cpu', weights_only=True)
faster_rcnn_nighttime.load_state_dict(checkpoint['model_state_dict'])

# # Classified motobike
# faster_rcnn_motorbike = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
# in_features = faster_rcnn_motorbike.roi_heads.box_predictor.cls_score.in_features
# faster_rcnn_motorbike.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
# checkpoint = torch.load('./checkpoint/epoch_10_model.pth')
# faster_rcnn_motorbike.load_state_dict(checkpoint['model_state_dict'])

# # Classified car
# faster_rcnn_car = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
# in_features = faster_rcnn_car.roi_heads.box_predictor.cls_score.in_features
# faster_rcnn_car.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
# checkpoint = torch.load('./checkpoint/epoch_7_model.pth')
# faster_rcnn_car.load_state_dict(checkpoint['model_state_dict'])

# Classified motorbike daytime
faster_rcnn_motorbike_day = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
in_features = faster_rcnn_motorbike_day.roi_heads.box_predictor.cls_score.in_features
faster_rcnn_motorbike_day.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
checkpoint = torch.load('./checkpoint/epoch_9_model.pth')
faster_rcnn_motorbike_day.load_state_dict(checkpoint['model_state_dict'])

yolov10_daytime = YOLO('./checkpoint/bestYOLOv11x45_50e_lr=0.0015.pt')

yolov10_nighttime = YOLO('./checkpoint/bestYOLOv10x50e.pt')

yolov10_daytime_ver2 = YOLO('./checkpoint/bestYOLOv10x_daytime_50.pt')

yolov8_daytime = YOLO('./checkpoint/bestYOLOv8x50edaytime.pt')

yolov8_nighttime = YOLO('./checkpoint/bestYOLOv8x_nighttime_6_100_batch_8.pt')

yolov5_nighttime = YOLO('./checkpoint/bestYOLOv5x6u_nighttime.pt')

yolov5_daytime = YOLO('./checkpoint/bestYOLOv5x6u_daytime_58_100.pt')

faster_rcnn_daytime.eval()
faster_rcnn_nighttime.eval()
retinanet_nighttime.eval()

day_dataset = TrafficDataset(imgs_day)
day_loader = DataLoader(day_dataset, batch_size=1, shuffle=False)
night_dataset = TrafficDataset(imgs_night)
night_loader = DataLoader(night_dataset, batch_size=1, shuffle=False)
dark_night_dataset = TrafficDataset(imgs_dark_night)
dark_night_loader = DataLoader(dark_night_dataset, batch_size=1, shuffle=False)
bright_night_dataset = TrafficDataset(imgs_bright_night)
bright_night_loader = DataLoader(bright_night_dataset, batch_size=1, shuffle=False)

def inference_faster_rcnn(img, model, detection_threshold=0.0, nms_thresh=0.1):
    img = img.to(device)
    model.eval()
    model.to(device)
    outputs = model(img)

    boxes = outputs[0]['boxes'].data.cpu().numpy()
    scores = outputs[0]['scores'].data.cpu().numpy()
    labels = outputs[0]['labels'].data.cpu().numpy()

    tensr_boxes = torch.tensor(boxes)
    tensr_scores = torch.tensor(scores)
    keep_indices = nms(tensr_boxes, tensr_scores, nms_thresh)

    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    labels = labels[keep_indices]

    boxes = boxes[scores >= detection_threshold]
    labels = labels[scores >= detection_threshold]
    scores = scores[scores >= detection_threshold]

    # Remove backgoung class
    labels -= 1

    return boxes, scores, labels

def inference_retinanet(img, model, detection_threshold=0.0, nms_thresh=0.1):
    img = img.to(device)
    model.eval()
    model.to(device)
    outputs = model(img)

    boxes = outputs[0]['boxes'].data.cpu().numpy()
    scores = outputs[0]['scores'].data.cpu().numpy()
    labels = outputs[0]['labels'].data.cpu().numpy()

    tensr_boxes = torch.tensor(boxes)
    tensr_scores = torch.tensor(scores)
    keep_indices = nms(tensr_boxes, tensr_scores, nms_thresh)

    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    labels = labels[keep_indices]

    boxes = boxes[scores >= detection_threshold]
    labels = labels[scores >= detection_threshold]
    scores = scores[scores >= detection_threshold]

    # Remove backgoung class
    labels -= 1

    return boxes, scores, labels

def convert_to_yolo(boxes, img_width=1280, img_height=720):
    boxes_convert = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        # Calculate width and height of bounding box
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Calculate center coordinates
        x_center = x_min + box_width / 2
        y_center = y_min + box_height / 2

        # Normalize by image dimensions
        x_center /= img_width
        y_center /= img_height
        box_width /= img_width
        box_height /= img_height
        box = [x_center, y_center, box_width, box_height]
        boxes_convert.append(box)
    return boxes_convert

def convert_ensemble_to_yolo(boxes):
    boxes_convert = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        # Calculate width and height of bounding box
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Calculate center coordinates
        x_center = x_min + box_width / 2
        y_center = y_min + box_height / 2

        box = [x_center, y_center, box_width, box_height]
        boxes_convert.append(box)
    return boxes_convert

def convert_coord_ensemble(boxes):
    boxes_convert = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        x_min, x_max = x_min/1280, x_max/1280
        y_min, y_max = y_min/720, y_max/720
        box = [x_min, y_min, x_max, y_max]
        boxes_convert.append(box)
    return boxes_convert

def convert_coord_original(boxes):
    boxes_convert = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        x_min, x_max = x_min*1280, x_max*1280
        y_min, y_max = y_min*720, y_max*720
        box = [x_min, y_min, x_max, y_max]
        boxes_convert.append(box)
    return boxes_convert

def inference_yolo(img, model, detection_threshold=0.0, nms_thresh=0.1):

    results = model(img)[0]
    detections = sv.Detections.from_ultralytics(results)
    bounding_boxes = results.boxes.xywhn.cpu().numpy()
    labels = detections.class_id
    scores = detections.confidence

    boxes = []

    for box in bounding_boxes:
        x_center, y_center, width, height = box
        xmin, ymin, xmax, ymax = convert_to_pixel_coords(x_center, y_center, width, height, 1280, 720)
        box = [xmin, ymin, xmax, ymax]
        boxes.append(box)

    boxes1 = []
    scores1 = []
    labels1 = []

    for i in range(len(boxes)):
        if scores[i] >= detection_threshold:
            boxes1.append(boxes[i])
            scores1.append(scores[i])
            labels1.append(labels[i])

    return boxes1, scores1, labels1

def ensemble_predictions_nighttime(image, image_origin, detection_thr=0.0, iou_thr=0.5, skip_box_thr=0.0, weights=[1, 1, 1, 1, 1], conf_type='avg'):
    # Get predictions from each model
    # boxes_faster_day, scores_faster_day, labels_faster_day = inference_faster_rcnn(image, faster_rcnn_daytime)
    boxes_faster_night, scores_faster_night, labels_faster_night = inference_faster_rcnn(image, faster_rcnn_nighttime)
    boxes_retina, scores_retina, labels_retina = inference_retinanet(image, retinanet_nighttime)
    boxes_yolov5, scores_yolov5, labels_yolov5 = inference_yolo(image_origin, yolov5_nighttime, detection_threshold=0.0)
    boxes_yolov8, scores_yolov8, labels_yolov8 = inference_yolo(image_origin, yolov8_nighttime, detection_threshold=0.0)
    boxes_yolov10, scores_yolov10, labels_yolov10 = inference_yolo(image_origin, yolov10_nighttime, detection_threshold=0.0)

    # Convert to coordinate of ensemble
    # boxes_faster_day = convert_coord_ensemble(boxes_faster_day)
    boxes_faster_night = convert_coord_ensemble(boxes_faster_night)
    boxes_retina = convert_coord_ensemble(boxes_retina)
    boxes_yolov5 = convert_coord_ensemble(boxes_yolov5)
    boxes_yolov8 = convert_coord_ensemble(boxes_yolov8)
    boxes_yolov10 = convert_coord_ensemble(boxes_yolov10)

    boxes_list = [boxes_faster_night, boxes_retina, boxes_yolov5, boxes_yolov8, boxes_yolov10]
    scores_list = [scores_faster_night, scores_retina, scores_yolov5, scores_yolov8, scores_yolov10]
    labels_list = [labels_faster_night, labels_retina, labels_yolov5, labels_yolov8, labels_yolov10]

    # Perform WBF
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type=conf_type
    )

    boxes_filter = []
    scores_filter = []
    labels_filter = []
    for i in range(len(boxes)):
        if scores[i] >= detection_thr:
            boxes_filter.append(boxes[i])
            scores_filter.append(scores[i])
            labels_filter.append(labels[i])

    # Convert to original coordinate
    boxes_filter = convert_coord_original(boxes_filter)

    return boxes_filter, scores_filter, labels_filter

def ensemble_prediction_daytime(image_tensor, image_origin, detection_thr=0.0, iou_thr=0.5, skip_box_thr=0.0, weights=[1, 1, 1, 1, 1], conf_type='avg'):
    image_origin_cvt = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
    # Get predictions from each model
    boxes_faster_day, scores_faster_day, labels_faster_day = inference_faster_rcnn(image_tensor, faster_rcnn_daytime, detection_threshold=0.0)
    boxes_yolov10, scores_yolov10, labels_yolov10 = inference_yolo(image_origin, yolov10_daytime)
    boxes_yolov8, scores_yolov8, labels_yolov8 = inference_yolo(image_origin, yolov8_daytime)
    boxes_yolov5, scores_yolov5, labels_yolov5 = inference_yolo(image_origin, yolov5_daytime)
    boxes_yolov10_ver2, scores_yolov10_ver2, labels_yolov10_ver2 = inference_yolo(image_origin, yolov10_daytime_ver2)

    # Convert to coordinate of ensemble
    boxes_faster_day = convert_coord_ensemble(boxes_faster_day)
    boxes_yolov10 = convert_coord_ensemble(boxes_yolov10)
    boxes_yolov8 = convert_coord_ensemble(boxes_yolov8)
    boxes_yolov5 = convert_coord_ensemble(boxes_yolov5)
    boxes_yolov10_ver2 = convert_coord_ensemble(boxes_yolov10_ver2)
    # boxes_faster_motorbike = convert_coord_ensemble(boxes_faster_motorbike)

    boxes_list = [boxes_faster_day, boxes_yolov10, boxes_yolov8, boxes_yolov5, boxes_yolov10_ver2]
    scores_list = [scores_faster_day, scores_yolov10, scores_yolov8, scores_yolov5, scores_yolov10_ver2]
    labels_list = [labels_faster_day, labels_yolov10, labels_yolov8, labels_yolov5, labels_yolov10_ver2]

    # Perform WBF
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type=conf_type
    )

    boxes_filter = []
    scores_filter = []
    labels_filter = []
    for i in range(len(boxes)):
        if scores[i] >= detection_thr:
            boxes_filter.append(boxes[i])
            scores_filter.append(scores[i])
            labels_filter.append(labels[i])

    # Convert to original coordinate
    boxes_filter = convert_coord_original(boxes_filter)

    return boxes_filter, scores_filter, labels_filter

def insert_prediction(img_path, labels, boxes, scores, filepath):
    with open(filepath, "a") as f:
        for i in range(len(boxes)):
            s = ""
            s += os.path.basename(img_path) + " "
            s += str(labels[i])[0] + " "
            for j in range(4):
                s += str(boxes[i][j]) + " "
            s += str(scores[i]) + "\n"
            f.write(s)
    f.close()
    
with open('./predict.txt', 'w') as f:
    pass

for i, image in tqdm(enumerate(night_loader)):
    image = image.to(device)
    image_origin = cv2.imread(os.path.join(folder_path, 'images', os.path.basename(img_night_paths[i])))

    boxes, scores, labels = ensemble_predictions_nighttime(image, image_origin, weights=[3, 1, 1, 1, 1])
    boxes = convert_to_yolo(boxes)
    insert_prediction(img_night_paths[i], labels, boxes, scores, './predict.txt')

for i, image in tqdm(enumerate(dark_night_loader)):
    image = image.to(device)
    # Original image
    origin_image = cv2.imread(os.path.join(folder_path, 'images', os.path.basename(img_dark_night_paths[i])))
    origin_image_cvt = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    origin_image_normed = trans(origin_image_cvt).unsqueeze(0)
    origin_image_normed = origin_image_normed.to(device)

    boxes_retinanet, scores_retinanet, labels_retinanet = inference_retinanet(origin_image_normed, retinanet_nighttime)
    boxes_retinanet = convert_coord_ensemble(boxes_retinanet)

    boxes_yolov5, scores_yolov5, labels_yolov5 = inference_yolo(origin_image, yolov5_nighttime)
    boxes_yolov5 = convert_coord_ensemble(boxes_yolov5)

    # Inhanced image
    boxes_fasterrcnn, scores_fasterrcnn, labels_fasterrcnn = inference_faster_rcnn(image, faster_rcnn_nighttime)
    boxes_fasterrcnn = convert_coord_ensemble(boxes_fasterrcnn)

    boxes_retinanet_enhanced, scores_retinanet_enhanced, labels_retinanet_enhanced = inference_retinanet(image, retinanet_nighttime)
    boxes_retinanet_enhanced = convert_coord_ensemble(boxes_retinanet_enhanced)

    # Ensemble
    boxes = [boxes_fasterrcnn, boxes_retinanet, boxes_retinanet_enhanced, boxes_yolov5]
    scores = [scores_fasterrcnn, scores_retinanet, scores_retinanet_enhanced, scores_yolov5]
    labels = [labels_fasterrcnn, labels_retinanet, labels_retinanet_enhanced, labels_yolov5]

    boxes, scores, labels = weighted_boxes_fusion(
        boxes, scores, labels, weights=[1, 1, 1, 1], iou_thr=0.5, skip_box_thr=0.0
    )

    boxes = convert_ensemble_to_yolo(boxes)
    insert_prediction(img_dark_night_paths[i], labels, boxes, scores, './predict.txt')

for i, image in tqdm(enumerate(bright_night_loader)):
    image = image.to(device)
    image_origin = cv2.imread(os.path.join(folder_path, 'images', os.path.basename(img_bright_night_paths[i])))

    boxes_faster_day, scores_faster_day, labels_faster_day = inference_faster_rcnn(image, faster_rcnn_daytime)
    boxes_faster_night, scores_faster_night, labels_faster_night = inference_faster_rcnn(image, faster_rcnn_nighttime)
    boxes_retina, scores_retina, labels_retina = inference_retinanet(image, retinanet_nighttime)
    boxes_yolo, scores_yolo, labels_yolo = inference_yolo(image_origin, yolov5_nighttime, detection_threshold=0.0)

    # Convert to coordinate of ensemble
    boxes_faster_day = convert_coord_ensemble(boxes_faster_day)
    boxes_faster_night = convert_coord_ensemble(boxes_faster_night)
    boxes_retina = convert_coord_ensemble(boxes_retina)
    boxes_yolo = convert_coord_ensemble(boxes_yolo)

    boxes_list = [boxes_faster_night, boxes_retina, boxes_yolo, boxes_faster_day]
    scores_list = [scores_faster_night, scores_retina, scores_yolo, scores_faster_day]
    labels_list = [labels_faster_night, labels_retina, labels_yolo, labels_faster_day]

    # Perform WBF
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, weights=[1.2, 0.8, 0.8, 1.2], iou_thr=0.5, skip_box_thr=0.0, conf_type='avg'
    )

    boxes = convert_ensemble_to_yolo(boxes)
    insert_prediction(img_bright_night_paths[i], labels, boxes, scores, './predict.txt')

for i, image in tqdm(enumerate(day_loader)):
    image = image.to(device)
    boxes, scores, labels = ensemble_prediction_daytime(image, imgs_day[i],detection_thr=0.0, weights=[1, 1, 1, 1, 1])
    boxes = convert_to_yolo(boxes)
    insert_prediction(filenames_day[i], labels, boxes, scores, './predict.txt')
    
