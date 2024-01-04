import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# keras
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def invert(image):
    return 255-image

def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
    plt.show()

def dilate(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)

# test za proveru rada 
test_resize_img = load_image('data1/pictures/captcha_1.jpg')
test_resize_ref = (28, 28)
test_resize_res = resize_region(test_resize_img).shape[0:2]
# print("Test resize passsed: ", test_resize_res == test_resize_ref)

def select_roi(image_orig, image_bin):
    image_orig = image_orig[160:400, 230:850]
    image_bin = image_bin[160:400, 230:850]
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 18))
    image_bin = invert(image_bin)
    threshed = cv2.morphologyEx(image_bin, cv2.MORPH_CLOSE, rect_kernel)
    contours, hierarchy = cv2.findContours(threshed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour) # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if area > 150 and h < 100 and h > 5 and w > 20:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # oznaciti region pravougaonikom na originalnoj slici sa rectangle funkcijom
            region = image_bin[y:y+h+1, x:x+w+1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    return image_orig, [region[0] for region in regions_array]

def select_roi_with_distances(image_orig, image_bin):
    image_orig = image_orig[160:400, 230:850]
    image_bin = image_bin[160:400, 230:850]
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 18))
    image_bin = invert(image_bin)
    threshed = cv2.morphologyEx(image_bin, cv2.MORPH_CLOSE, rect_kernel)
    contours, hierarchy = cv2.findContours(threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        region = image_bin[y:y+h+1, x:x+w+1]
        regions_array.append([resize_region(region), (x, y, w, h)])
        cv2.rectangle(image_orig, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    
    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])
        region_distances.append(distance)
    
    return image_orig, sorted_regions, region_distances

def scale_to_range(image):
    return image/255

# test za proveru
test_scale_matrix = np.array([[0, 255], [51, 153]], dtype='float')
test_scale_ref = np.array([[0., 1.], [0.2, 0.6]], dtype='float')
test_scale_res = scale_to_range(test_scale_matrix)
# print("Test scale passed: ", np.array_equal(test_scale_res, test_scale_ref))

def matrix_to_vector(image):
    return image.flatten()

test_mtv = np.ndarray((28, 28))
test_mtv_ref = (784, )
test_mtv_res = matrix_to_vector(test_mtv).shape
# print("Test matrix to vector passed: ", test_mtv_res == test_mtv_ref)

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann

def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

# test konverzije
test_convert_alphabet = [0, 1, 2]
test_convert_ref = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='float')
test_convert_res = convert_output(test_convert_alphabet).astype('float')
# print("Test convert output: ", np.array_equal(test_convert_res, test_convert_ref))

def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann

def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32) # dati ulaz
    y_train = np.array(y_train, np.float32) # zeljeni izlazi na date ulaze
    
    # print("\nTraining started...")
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0, shuffle=False)
    # print("\nTraining completed...")
    return ann

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

test_winner_output = [0., 0.2, 0.3, 0.95]
test_winner_ref = 3
test_winner_res = winner(test_winner_output)
# print("Test winner passed: ", test_winner_res == test_winner_ref)

def display_result_with_spaces(outputs, ground_truth, alphabet, region_distances):
    n = 0
    i = 0
    for distance in region_distances:
        n += distance
        i += 1
    average_distance = n/i
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:, :]):
        if region_distances[idx] > average_distance * 1.7:
            result += ' '
        result += alphabet[winner(output)]
    hamming_distance = hammingDist(result, ground_truth)
    return result, hamming_distance

def hammingDist(str1, str2):
    i = 0
    count = 0
    if len(str1) != len(str2):
        print("Input strings must have equal length!")
    for char in str2:
        if (char != str1[i]):
            count+=1
        i+=1
    return count

image_folder = sys.argv[1] + 'pictures/'
csv_path = sys.argv[1] + 'res.csv'
df = pd.read_csv(csv_path)
images = df['file']
solutions = df['text']
alphabet = []
regions = []
error = 0
for i, solution in enumerate(solutions):
    image_file = images[i]
    image_path = os.path.join(image_folder, image_file)
    image_color = load_image(image_path)
    img_bin = image_bin(image_gray(image_color))
    selected_regions, letters = select_roi(image_color.copy(), img_bin)
    solution = solution.replace(' ','')
    for j, letter in enumerate(solution):
        if letter == ' ':
            continue
        if letter not in alphabet:
            alphabet.append(letter)
            regions.append(letters[j])


inputs = prepare_for_ann(regions)
outputs = convert_output(alphabet)
ann = create_ann(output_size=len(alphabet))
ann = train_ann(ann, inputs, outputs, epochs=1000)

for i, image in enumerate(images):
    image_path = os.path.join(image_folder, image)
    image_color = load_image(image_path)
    img_bin = image_bin(image_gray(image_color))
    selected_regions, letters, distances = select_roi_with_distances(image_color.copy(), img_bin)

    inputs = prepare_for_ann(letters)
    results = ann.predict(np.array(inputs, np.float32), verbose=0)
    ground_truth = solutions[i]
    recognized_text, hamming_distance = display_result_with_spaces(results, ground_truth, alphabet, distances)
    error += hamming_distance
    print(f"{image}-{ground_truth}-{recognized_text}")

print("Hamming distance: ", error)