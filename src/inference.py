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
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


device = torch.device("cpu")

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((720, 1280))
    ])
def load_fasterrcnn_model(path, v2=False):
    if v2:
        model = fasterrcnn_resnet50_fpn_v2(num_classes=5)
    else:
        model = fasterrcnn_resnet50_fpn(num_classes=5)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def load_retinanet_model(path):
    model = retinanet_resnet50_fpn(progress=True)
    in_features = list(model.head.classification_head.conv)[0][0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_features,
        num_anchors=num_anchors,
        num_classes=5
    )
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint)
    return model

retinanet_nighttime = load_retinanet_model('./checkpoint/Retinanet_nighttime.pt')
faster_rcnn_daytime = load_fasterrcnn_model('./checkpoint/faster_rcnn_day.pth')
faster_rcnn_nighttime = load_fasterrcnn_model('./checkpoint/faster_rcnn_night.pth', v2=True)
yolo11_daytime = YOLO('checkpoint/best_yolo11x_all_ep50.pt')
yolov10_nighttime = YOLO('checkpoint/best_yolov10x_all_ep50.pt')
yolov10_daytime = YOLO('checkpoint/best_yolov10x_day_ep50.pt')
yolov8_daytime = YOLO('checkpoint/best_yolov8x_day_ep50.pt')
yolov8_nighttime = YOLO('checkpoint/best_yolov8x_night_ep8.pt')
yolov5_nighttime = YOLO('checkpoint/best_yolov5x6u_night_ep60.pt')
yolov5_daytime = YOLO('checkpoint/best_yolov5x6u_day_ep60.pt')

print("Load all models successfully")

faster_rcnn_daytime.eval()
faster_rcnn_nighttime.eval()
retinanet_nighttime.eval()


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


class TrafficDataset(Dataset):
    def __init__(self, image_paths, trans=None):
        self.image_paths = image_paths
        self.trans = trans

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.trans:
            image = self.trans(image)

        return image

dataset_path = 'dataset/original'
ptestpath = f'{dataset_path}/public test'

img_paths = []

for img_name in os.listdir(ptestpath):
    img_paths.append(os.path.join(ptestpath, img_name))
filenames_day = []
filenames_night = []

for img_path in img_paths:
    if 'cam_10' in img_path or 'src_1_frame' in img_path:
        filenames_day.append(img_path)
    else:
        filenames_night.append(img_path)
    
day_img_paths = []
night_img_paths = []

for img_path in filenames_day:
    day_img_paths.append(img_path)

for img_path in filenames_night:
    night_img_paths.append(img_path)
    
img_dark_night_paths = []
img_night_paths = []

with open("secret.pkl", "rb") as f:
    a = pickle.load(f)
    need_enhanced_image = a['need_enhanced_image']

for filename_path in sorted(filenames_night):
    if need_enhanced_image in os.path.basename(filename_path):
        img_dark_night_path = os.path.join('dataset/enhanced_images', os.path.basename(filename_path))
        img_dark_night_paths.append(img_dark_night_path)
    else:
        img_night_path = os.path.join('dataset/images', os.path.basename(filename_path))
        img_night_paths.append(img_night_path)

img_bright_night_paths = img_dark_night_paths[261:]
img_dark_night_paths = img_dark_night_paths[:261]

imgs_dark_night = []
imgs_bright_night = []
night_img_paths = []

for img_path in img_dark_night_paths:
    imgs_dark_night.append(img_path)

for img_path in img_bright_night_paths:
    imgs_bright_night.append(img_path)

for img_path in img_night_paths:
    night_img_paths.append(img_path)

day_dataset = TrafficDataset(day_img_paths, trans)
day_loader = DataLoader(day_dataset, batch_size=1, shuffle=False)
night_dataset = TrafficDataset(night_img_paths, trans)
night_loader = DataLoader(night_dataset, batch_size=1, shuffle=False)
dark_night_dataset = TrafficDataset(imgs_dark_night, trans)
dark_night_loader = DataLoader(dark_night_dataset, batch_size=1, shuffle=False)
bright_night_dataset = TrafficDataset(imgs_bright_night, trans)
bright_night_loader = DataLoader(bright_night_dataset, batch_size=1, shuffle=False)

def inference_faster_rcnn(img, model, detection_threshold=0.0, nms_thresh=0.1):
    img = img.to(device)
    model.to(device)
    with torch.no_grad():
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
    model.to(device)

    with torch.no_grad():
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
    results = model(img, verbose=False)[0]
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
    boxes_yolov10, scores_yolov10, labels_yolov10 = inference_yolo(image_origin, yolo11_daytime)
    boxes_yolov8, scores_yolov8, labels_yolov8 = inference_yolo(image_origin, yolov8_daytime)
    boxes_yolov5, scores_yolov5, labels_yolov5 = inference_yolo(image_origin, yolov5_daytime)
    boxes_yolov10_ver2, scores_yolov10_ver2, labels_yolov10_ver2 = inference_yolo(image_origin, yolov10_daytime)

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

# for i, image in tqdm(enumerate(night_loader)):
#     image_origin = cv2.imread(os.path.join('dataset/images', os.path.basename(img_night_paths[i])))
#     boxes, scores, labels = ensemble_predictions_nighttime(image, image_origin, weights=[3, 1, 1, 1, 1])
#     boxes = convert_to_yolo(boxes)
#     insert_prediction(img_night_paths[i], labels, boxes, scores, './predict.txt')

for i, image in tqdm(enumerate(dark_night_loader)):
    image = image.to(device)
    origin_image = cv2.imread(os.path.join('dataset/images', os.path.basename(img_dark_night_paths[i])))
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

# for i, image in tqdm(enumerate(bright_night_loader)):
#     image = image.to(device)
#     image_origin = cv2.imread(os.path.join('dataset/images', os.path.basename(img_bright_night_paths[i])))

#     boxes_faster_day, scores_faster_day, labels_faster_day = inference_faster_rcnn(image, faster_rcnn_daytime)
#     boxes_faster_night, scores_faster_night, labels_faster_night = inference_faster_rcnn(image, faster_rcnn_nighttime)
#     boxes_retina, scores_retina, labels_retina = inference_retinanet(image, retinanet_nighttime)
#     boxes_yolo, scores_yolo, labels_yolo = inference_yolo(image_origin, yolov5_nighttime, detection_threshold=0.0)

#     # Convert to coordinate of ensemble
#     boxes_faster_day = convert_coord_ensemble(boxes_faster_day)
#     boxes_faster_night = convert_coord_ensemble(boxes_faster_night)
#     boxes_retina = convert_coord_ensemble(boxes_retina)
#     boxes_yolo = convert_coord_ensemble(boxes_yolo)

#     boxes_list = [boxes_faster_night, boxes_retina, boxes_yolo, boxes_faster_day]
#     scores_list = [scores_faster_night, scores_retina, scores_yolo, scores_faster_day]
#     labels_list = [labels_faster_night, labels_retina, labels_yolo, labels_faster_day]

#     # Perform WBF
#     boxes, scores, labels = weighted_boxes_fusion(
#         boxes_list, scores_list, labels_list, weights=[1.2, 0.8, 0.8, 1.2], iou_thr=0.5, skip_box_thr=0.0, conf_type='avg'
#     )

#     boxes = convert_ensemble_to_yolo(boxes)
#     insert_prediction(img_bright_night_paths[i], labels, boxes, scores, './predict.txt')

# for i, image in tqdm(enumerate(day_loader)):
#     image = image.to(device)
#     boxes, scores, labels = ensemble_prediction_daytime(image, day_img_paths[i],detection_thr=0.0, weights=[1, 1, 1, 1, 1])
#     boxes = convert_to_yolo(boxes)
#     insert_prediction(filenames_day[i], labels, boxes, scores, './predict.txt')