import numpy as np
import cv2
import os
import zipfile
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

from tqdm import tqdm

classnames = ['daytime', 'nighttime']
le = LabelEncoder()
le.fit_transform(classnames)
print(le.classes_)
print(le.transform(le.classes_))

folder = 'daytime'
img_paths = []
label = 0           # daytime

current_directory = os.getcwd()
daytime_path = os.path.join(current_directory, folder)

for img_name in sorted(os.listdir(daytime_path))[:8000]:
    if img_name.endswith('.jpg'):
        img_path = os.path.join(daytime_path, img_name)
        img_paths.append(img_path)
    else:
        continue

# Read the dataset
img_datas = []
for img_path in tqdm(img_paths):
    img = cv2.imread(img_path)
    img_datas.append([img_path, img, label])
    
folder = 'nighttime'
img_paths = []
label = 1                   # night time

current_directory = os.getcwd()
nighttime_path = os.path.join(current_directory, folder)

for img_name in os.listdir(nighttime_path):
    if img_name.endswith('.jpg'):
        img_path = os.path.join(nighttime_path, img_name)
        img_paths.append(img_path)
    else:
        continue

# Read the dataset
for img_path in tqdm(img_paths):
    img = cv2.imread(img_path)
    img_datas.append([img_path, img, label])
    
# Compute CDF
def compute_cdf(img):
    # convert to gray image
    if len(img) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # compute the histogram
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    
    # compute the cdf
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()
    return cdf_normalized

def compute_hist(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    hist = hist / hist.max()
    
    return hist

data = [img_datas[i][1] for i in range(len(img_datas))]
labels = [img_datas[i][2] for i in range(len(img_datas))]

dataset = []

for i in tqdm(range(len(data))):
    img_cdf = compute_hist(data[i])
    dataset.append(img_cdf)
    
estimators = [
    ('rf_model', RandomForestClassifier(n_estimators=1000, random_state=42)),
    ('lg_model', LogisticRegression())
]

# model
X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=0.2, shuffle=True, random_state=42)

clf = StackingClassifier(estimators=estimators, final_estimator=SVC(kernel='linear', probability=True))
clf.fit(X_train, Y_train)

with open('./checkpoint/stacking_model_weights_histogram.pkl', 'wb') as file:
    pickle.dump(clf, file)