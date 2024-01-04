import numpy as np
import cv2
import sys
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import threading


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def non_max_suppression_slow(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    
    pick = []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    score = boxes[:, 4]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(score)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        for pos in range(0, last):
            j = idxs[pos]
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            overlap = float(w * h) / area[j]

            if overlap > overlapThresh:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)

    return pick

def get_hog():
    img_size = (60,120)
    nbins = 9
    cell_size = (6, 6)
    block_size = (2, 2)
    hog = cv2.HOGDescriptor(_winSize=(img_size[0] // cell_size[0] * cell_size[0],
                                      img_size[1] // cell_size[1] * cell_size[1]),
                            _blockSize=(block_size[0] * cell_size[0],
                                        block_size[1] * cell_size[1]),
                            _blockStride=(cell_size[0], cell_size[1]),
                            _cellSize=(cell_size[0], cell_size[1]),
                            _nbins=nbins)
    return hog

def train_classifier(hog, pos_img, neg_img):
    pos_features = []
    neg_features = []
    labels = []

    for img in pos_img:
        pos_features.append(hog.compute(img))
        labels.append(1)

    for img in neg_img: 
        neg_features.append(hog.compute(img)) 
        labels.append(0)

    pos_features = np.array(pos_features)
    neg_features = np.array(neg_features)
    x = np.vstack((pos_features, neg_features))
    y = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=426)
    clf_svm = SVC(kernel='linear', probability=True) 
    clf_svm.fit(x_train, y_train)
    y_train_pred = clf_svm.predict(x_train)
    y_test_pred = clf_svm.predict(x_test)
    print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
    print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))
    return clf_svm

def detect_line(img):
    gray_img = img
    edges_img = cv2.Canny(gray_img, 80, 150, apertureSize=3)
    
    min_line_length = 200
    
    lines = cv2.HoughLinesP(image=edges_img, rho=1, theta=np.pi/180, threshold=10, lines=np.array([]),
                            minLineLength=min_line_length, maxLineGap=140)
    
    img_with_lines = img.copy()
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        Angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi 
        if abs(Angle) < 1:
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 20)  # (0, 255, 0) is the color (green), and 2 is the line thickness
            break
    y1 = 2160 - y1     
    y2 = 2160 - y2
    
    return x1, y1, x2, y2

def get_line_params(line_coords):
    k = (float(line_coords[3]) - float(line_coords[1])) / (float(line_coords[2]) - float(line_coords[0]))
    n = k * (float(-line_coords[0])) + float(line_coords[1])
    return k, n

def detect_cross(x, y, k, n):
    yy = k*x + n
    return  abs(yy - y) <= 30

def classify_window(window):
    window = cv2.resize(window, (60, 120), interpolation=cv2.INTER_NEAREST)
    features = hog.compute(window).reshape(1, -1)
    return classifier.predict_proba(features)[0][1]

def process_image(image, step_size= 15, window_size=(80, 160)):
    bounding_boxes=[]
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            window = image[y:y+window_size[1],x:x+window_size[0]]
            if window.shape == (window_size[1], window_size[0], 3):
                score = classify_window(window)
                if score > 0.95:
                    bounding_boxes.append((x, y, x+window_size[0], y+window_size[1],score))

    pick = non_max_suppression_slow(np.array(bounding_boxes),0.3)
    selected_rectangles = [bounding_boxes[i] for i in pick]
    return selected_rectangles


def process_video(video_path):
    sum_of_nums = 0
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num)

    grabed, frame = cap.read()
    frame = frame[890:1300, 1410:2450]
    line_coords = detect_line(frame)
    k, n = get_line_params(line_coords)
    n = 2160 - n 
    
    # izdvajanje krajnjih x koordinata linije
    line_left_x = line_coords[0]
    line_right_x = line_coords[2]
    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break
        frame = frame[850:1300, 1410:2500]
        if frame_num == 0: # ako je prvi frejm, detektuj liniju
            line_coords = detect_line(frame)
            k, n = get_line_params(line_coords)
            n = 2160 - n
            line_left_x = line_coords[0]
            line_right_x = line_coords[2]

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        rectangles = process_image(frame)
        image_with_rectangles = frame.copy()
        for box in rectangles:
            x1, y1, x2, y2,score = box
            cv2.rectangle(image_with_rectangles, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for rectangle in rectangles:
            x1, y1, x2, y2, score = rectangle
            w=abs(x1-x2)
            h=abs(y1-y2)
            centar_y = y1 + h /2
            if abs(centar_y - n) < 15:
                sum_of_nums += 1 
        frame_num += 2  
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_num)
    cap.release()
    return sum_of_nums

image_dir = sys.argv[1] + 'pictures/'
pos_img = []
neg_img = []

for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    img = load_image(img_path)
    if 'p_' in img_name:
        pos_img.append(img)
    elif 'n_' in img_name:
        neg_img.append(img)

hog = get_hog()
classifier = train_classifier(hog, pos_img, neg_img)

csv_path = sys.argv[1] + 'counts.csv'
video_path = sys.argv[1]+'videos'
images_path = sys.argv[1]+'pictures'

df = pd.read_csv(csv_path)
kolizija_column = df['Broj_prelaza']
video_files = [f for f in os.listdir(video_path) if f.endswith('.mp4')]
actual_values = []
predicted_values = []
absolute_diff = []

def threadWrapper(kolizija_correct,video_name, hog, classifier):
    kolizija_predict = process_video(video_name)
    predicted_values.append(kolizija_predict)
    absolute_diff.append(abs(kolizija_predict-kolizija_correct))
    video = video_name.split('/')[2]
    print(f'{video}-{kolizija_correct}-{kolizija_predict}')

threads = []
for video in video_files:
    kolizija_correct = df.loc[df['Naziv_videa'] == video, 'Broj_prelaza'].values[0]
    actual_values.append(kolizija_correct)
    video_name = video_path + '/' + video
    t = threading.Thread(target=threadWrapper,args=(kolizija_correct,video_name,hog,classifier))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

i = 0
error = 0
for diff in absolute_diff:
    i += 1
    error += diff
mae = error/i

print("MAE: ",mae)