import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.ops import nms

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
classes = ["motorbike", "car", "bus", "container"]

def load_model(path):
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    NUM_CLASSES = len(classes) + 1 # add background class

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

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def inference(img, model, detection_threshold=0.7, nms_thresh=0.1):
  img = img.to(device)
  outputs = model([img])

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

  return boxes, scores, labels