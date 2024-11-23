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
    model = YOLO('yolov10x.pt')

    # Set hyperparameters for fine-tuning
    model.train(
        data="./dataset/yolo/day/config.yaml",
        epochs=50,
        imgsz=640,
        seed=SEED,
        batch=16,
    )
    model.save("./20million_vehicle_detection/checkpoints/bestYOLOv10x_daytime_50.pt")


if __name__ == "__main__":
    main()