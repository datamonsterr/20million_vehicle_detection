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
    # Initialize the parser
    parser = argparse.ArgumentParser(description="DETR model object detection script")

    # Define command-line arguments
    parser.add_argument('--model_name', type=str, required=True, help="Name of yolo model")
    parser.add_argument('--data_config', type=str, default="configs/vehicles_dataset_yolo.yaml", help="URL of the image for object detection")
    parser.add_argument('--save_path', type=str, required=True, help="Directory to save the results")
    parser.add_argument('--batch_size', type=int, default=16, help="Set batch size")
    parser.add_argument('--freeze', type=int, default=None, help="Freeze from layer 1 to layer N")

    # Parse the arguments
    args = parser.parse_args()

    # Load the YOLO model
    model = YOLO(f"{args.model_name}.pt")

    # Set hyperparameters for fine-tuning
    model.train(
        data=args.data_config,  # Path to your dataset
        epochs=30,               # Increase the number of epochs for better convergence
        imgsz=768,               # Increase image size for more details
        seed=SEED,
        batch=args.batch_size,
        freeze=args.freeze,
    )
    model.save(args.save_path)

if __name__ == "__main__":
    main()
