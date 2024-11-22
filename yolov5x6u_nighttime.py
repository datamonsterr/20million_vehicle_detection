import random
from pathlib import Path
import glob
import os
from ultralytics import YOLO
import torch
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    SEED = 42
    set_seed(SEED)

    # Load the YOLO model
    model = YOLO('yolov5x6u.pt')

    # Set hyperparameters for fine-tuning
    model.train(
        data="./dataset/yolo/night/config.yaml",
        epochs=12,
        imgsz=1280,
        seed=SEED,
        batch=4,
        close_mosaic=0
    )
    model.save("./20million_vehicle_detection/checkpoints/bestYOLOv5x6u_nighttime.pt")


if __name__ == "__main__":
    main()