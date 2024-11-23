from ultralytics import YOLO
import argparse
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
    parser = argparse.ArgumentParser(description="DETR model object detection script")

    parser.add_argument('--model_name', type=str, required=True, help="Name of yolo model")
    parser.add_argument('--epoch', type=int, default=60, help="Number of epochs")
    parser.add_argument('--dataset', type=str, default="all", help="day | night | all")
    parser.add_argument('--batch_size', type=int, default=16, help="Set batch size")
    parser.add_argument('--freeze', type=int, default=None, help="Freeze from layer 1 to layer N")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0.937, help="Learning rate momentum")

    args = parser.parse_args()

    model = YOLO(f"{args.model_name}.pt")

    model.train(
        data=f"./dataset/yolo/{args.dataset}/config.yaml",  
        epochs=args.epoch,              
        imgsz=1280,               
        batch=args.batch_size,
        freeze=args.freeze,
        seed=SEED,
        lr0 = args.lr,
        momentum = args.momentum,
    )
    model.save(f"./checkpoint/best_{args.model_name}_{args.dataset}_ep{args.epoch}.pt")

if __name__ == "__main__":
    main()
