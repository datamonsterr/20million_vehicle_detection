import os
import shutil
import random
import argparse
import gdown

import os

def download_dataset_from_drive(id, data_path, desc):
    gdrive_url = f"https://drive.google.com/uc?id={id}"

    output_path = "temp.zip"

    print(f"Downloading the {desc} file...")
    gdown.download(gdrive_url, output_path, quiet=False)

    print("Unzipping the file...")
    unzip_path = data_path
    os.makedirs(unzip_path, exist_ok=True)  # Ensure the output folder exists

    shutil.unpack_archive(output_path, unzip_path, "zip")

    print(f"File unzipped to {unzip_path}")

    os.remove(output_path)
    print("Cleanup: Removed the downloaded zip file.")
    
def relabel_nighttime():
    label_map = {4: 0, 5: 1, 6: 2, 7: 3}
    dataset = "dataset/original/train/nighttime"
    for file in os.listdir(dataset):
        if file.endswith(".txt"):
                file_path = os.path.join(dataset, file)
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                modified_lines = []
                for line in lines:
                    parts = line.strip().split()
                    label = int(parts[0])
                    # Thay thế nhãn nếu thuộc các nhãn 4-7
                    if label in label_map:
                        parts[0] = str(label_map[label])
                    modified_lines.append(" ".join(parts))

                # Ghi lại nội dung đã thay đổi vào file .txt
                with open(file_path, 'w') as file:
                    file.write("\n".join(modified_lines))

    print("Labels have been successfully merged in all .txt files!")

def yolo_yaml(path):
    content = f"""path: ../{path}

train: images/train
val: images/val

names:
    0: motorbike
    1: car
    2: bus
    3: container
"""
    return content

def restructure_to_yolo(daytime=True, nighttime=True, original_path="dataset/original/train"):
    dataset_paths = []
    base_output_dir = 'dataset/yolo'

    if daytime and nighttime:
        output_dir = os.path.join(base_output_dir, 'all')
        dataset_paths.append(os.path.join(original_path, "daytime"))
        dataset_paths.append(os.path.join(original_path, "nighttime"))
    elif daytime:
        output_dir = os.path.join(base_output_dir,'day')
        dataset_paths.append(os.path.join(original_path, "daytime"))
    elif nighttime:
        output_dir = os.path.join(base_output_dir,'night')
        dataset_paths.append(os.path.join(original_path, "nighttime"))

    train_img_dir = os.path.join(output_dir, 'images/train')
    val_img_dir = os.path.join(output_dir, 'images/val')
    train_lbl_dir = os.path.join(output_dir, 'labels/train')
    val_lbl_dir = os.path.join(output_dir, 'labels/val')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        f.write(yolo_yaml(output_dir))

    train_ratio = 0.8

    for base_dir in dataset_paths:
        images = [f for f in os.listdir(base_dir) if f.endswith('.jpg')]
        random.shuffle(images) 

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
    # print("Downloading dataset ....")
    # download_dataset_from_drive("1SjMOqzzKDtmkqmiesIyDy2zkEN7xGjbE", "dataset/original/train", "training dataset")
    # download_dataset_from_drive("1BQvwhSoeDm-caCImtlbcAMzhI8MDsrCZ", "dataset/original/", "testing dataset")
    # print("Finished downloading dataset")
    relabel_nighttime()
    print("Restructuring dataset to YOLO format ....")
    restructure_to_yolo(daytime=True, nighttime=True)
    restructure_to_yolo(daytime=True, nighttime=False)
    restructure_to_yolo(daytime=False, nighttime=True)
    print("Finished restructuring dataset")
    
if __name__ == "__main__":
    main()