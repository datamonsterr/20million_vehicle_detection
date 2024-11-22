import argparse
import os
import pickle
import random
import torch
import torchvision
import cv2
import numpy as np
from PIL import Image

SEED = 42

def load_train_data(train_path):
    annotation_paths = {'day': [], 'night': []}
    img_paths = {'day': [], 'night': []}
    for t in ['day', 'night']:
        for root, _, files in os.walk(f"{train_path}/{t}time"):
            for file in files:
                if file.endswith(".jpg"):
                    if os.path.exists(os.path.join(root, file.replace(".jpg", ".txt"))):
                        img_paths[t].append(os.path.join(root,file))
                        annotation_paths[t].append(os.path.join(root, file.replace(".jpg", ".txt")))
    return img_paths, annotation_paths
        

def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    
if __name__ == "__main__":
    main()