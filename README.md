Upload models in [here](https://husteduvn-my.sharepoint.com/:f:/g/personal/dat_pt226024_sis_hust_edu_vn/EkxJJhwdG2VHrZgkkcqK02kBFEWw5Fhspqvf_gp9J_4kbg?e=1JmQTY)

# Dataset structure
- Original dataset will be store at: `dataset/original` with the same structure given by host.
- Some other dataset will be only processed or restructured image of the original one (Eg: `dataset/yolo`)

# Enhanced images using Zero DCE and Gamma Correction
- check out [this notebook](https://www.kaggle.com/code/hongmnhcng/train-zero-zero-reference)
- Running this notebook will give us `zeroref_enhanced images` and `gamma_correction_images`

# Reproduce
1. Fine-Tuning model RetinaNet NightTime
```bash
python src/trainers/retina_night.py
```
2. Fine-Tuning model Faster R-CNN DayTime
```bash
python src/trainers/fasterrcnn.py --dataset=day --epoch=50 --lr=0.0005 --batch_size=16 --num_workers=4
```
3. Fine-Tuning model Faster R-CNN NightTime
```bash
python src/trainers/fasterrcnn.py --dataset=night --epoch=50 --lr=0.0005 --batch_size=16 --num_workers=4
```
4. Fine-Tuning model YOLOv11x all
```bash
python src/trainers/yolo.py --model=yolo11x --dataset=all --epoch=50 --batch_size=16
```
5. Fine-Tuning model YOLOv10x all
```bash
python src/trainers/yolo.py --model=yolov10x --dataset=all --epoch=50 --batch_size=16
```
6. Fine-Tuning model YOLOv10x NightTime
```bash
python src/trainers/yolo.py --model=yolov10x --dataset=night --epoch=50 --batch_size=16
```
7. Fine-Tuning model YOLOv8x DayTime
```bash
python src/trainers/yolo.py --model=yolov8x --dataset=day --epoch=50 --batch_size=16
```
8. Fine-Tuning model YOLOv8x NightTime
```bash
python src/trainers/yolo.py --model=yolov8x --dataset=night --epoch=12 --batch_size=16
```
9. Fine-Tuning model YOLOv5x6u DayTime
```bash
python src/trainers/yolo.py --model=yolov5x6u --dataset=day --epoch=60 --batch_size=16
```
10. Fine-Tuning model YOLOv5x6u NightTime
```bash
python src/trainers/yolo.py --model=yolov5x6u --dataset=night --epoch=60 --batch_size=16
```

---

Inference result and save to `./predict.txt`:
```bash
python src/inference.py
```
