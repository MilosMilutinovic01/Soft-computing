import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return image_bin

def invert(image):
    return 255-image

def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
    plt.show()

# Define the folder path
folder_path = 'pictures1/'

# Get a list of all files in the folder
img_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.jpg', '.png', '.jpeg'))]

# Extract the numbers from the filenames for sorting
img_files = sorted(img_files, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))

# Load the images
images = [cv2.imread(img_path) for img_path in img_files]

actual = [4, 8, 6, 8, 8, 4, 6, 6, 6, 13] 
# ...

i = 0
sum = 0.0
result = 0

for img_path in img_files:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bin = image_bin(gray)
    kernel = np.ones((6,6), np.uint8)
    opening = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 0)
    sure_bg = cv2.dilate(dist_transform, kernel, iterations=5)
    img_tr = sure_bg < 10
    img_tr = invert(img_tr)
    sure_bg = img_tr
    
    ret, sure_fg = cv2.threshold(dist_transform, 0.05 * dist_transform.max(), 255, 0) 
    sure_fg = np.uint8(sure_fg)

    if len(sure_fg.shape) == 3:
        sure_fg = cv2.cvtColor(sure_fg, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(sure_fg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        #print(f"Contour Area: {area}")

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(h) / w

        if area > 49 and area < 650 and aspect_ratio > 0.7:
            valid_contours.append(contour)

    object_count = len(valid_contours)
    file_name = os.path.basename(img_path)
    print(f"{file_name}-{object_count}-{actual[i]}")
    sum += abs(actual[i] - object_count)
    i += 1

    # cv2.drawContours(img, valid_contours, -1, (0, 255, 0), 2)
    # cv2.imshow('Result', img)
    # cv2.waitKey(0)

cv2.destroyAllWindows()

error = sum/10
print(f"MAE: {error}")