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
    model = YOLO('yolov8x.pt')

    # Set hyperparameters for fine-tuning
    model.train(
        data="./dataset/yolo/night/config.yaml",
        epochs=6,
        imgsz=1280,
        seed=SEED,
        batch=8,
        close_mosaic=0
    )
    model.save("./20million_vehicle_detection/checkpoints/bestYOLOv8x_nighttime_6_100_batch_8.pt")


if __name__ == "__main__":
    main()