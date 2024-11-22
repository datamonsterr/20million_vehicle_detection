import os
import shutil
import random
import argparse

def restructure_to_yolo(daytime=True, nighttime=True):
    if dataset_paths:
        dataset_paths = [
            "dataset/original/train/daytime",
            "dataset/original/train/nighttime"
        ]

    output_dir = 'dataset/yolo'

    train_img_dir = os.path.join(output_dir, 'images/train')
    val_img_dir = os.path.join(output_dir, 'images/val')
    train_lbl_dir = os.path.join(output_dir, 'labels/train')
    val_lbl_dir = os.path.join(output_dir, 'labels/val')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    train_ratio = 0.8

    for base_dir in dataset_paths:
        images = [f for f in os.listdir(base_dir) if f.endswith('.jpg')]
        random.shuffle(images)  # Xáo trộn dữ liệu

        train_images = images[:int(len(images) * train_ratio)]
        val_images = images[int(len(images) * train_ratio):]

        for img_file in train_images:
            img_path = os.path.join(base_dir, img_file)
            label_file = img_file.replace('.jpg', '.txt')
            label_path = os.path.join(base_dir, label_file)

            shutil.copy(img_path, train_img_dir)
            if os.path.exists(label_path):
                shutil.copy(label_path, train_lbl_dir)  

        for img_file in val_images:
            img_path = os.path.join(base_dir, img_file)
            label_file = img_file.replace('.jpg', '.txt')
            label_path = os.path.join(base_dir, label_file)

            shutil.copy(img_path, val_img_dir)
            if os.path.exists(label_path):
                shutil.copy(label_path, val_lbl_dir)  

def main():
    parser = argparse.ArgumentParser(description="Download dataset")
    
    parser.add_argument("--yolo-daytime", type=bool, default=True, dest="restructure")
    
if __name__ == "__main__":
    main()