import pickle
import os

# def split_img_paths(img_paths):
#     day, night, dark, bright = [], [], [], []
#     for img_path in img_paths:
#         fname = os.path.basename(img_path)
#         if 'cam_10' in fname or 'src_1_frame' in fname:
#             day.append(fname)
#         else:
#             night.append(fname)
    
#     new_night = []
        
#     for fname in sorted(night):
#         if "src_2_frame" in fname:
#             dark.append(fname)
#         else:
#             new_night.append(fname)

#     night = new_night
#     bright = dark[261:]
#     dark = dark[:261]

#     print(f"Day: {len(day)}")
#     return day, night, dark, bright

# with open('model.pkl', 'wb') as f:
#     pickle.dump({"checkpoint": split_img_paths}, f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
    len(model["checkpoint"])