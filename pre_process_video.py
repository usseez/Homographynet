import os
import pickle
import random

import cv2 as cv
import numpy as np
from numpy.linalg import inv
from tqdm import tqdm

from config_video import image_folder
from config_video import train_file, valid_file, test_file


def get_datum(img, test_image, size, rho, top_point, patch_size):
    BL = (top_point[0], patch_size + top_point[1])
    BR = (patch_size + top_point[0], patch_size + top_point[1])
    TR = (patch_size + top_point[0], top_point[1])
    four_points = [top_point, BL, BR, TR]
    # print('top_point: ' + str(top_point))
    # print('left_point: ' + str(left_point))
    # print('bottom_point: ' + str(bottom_point))
    # print('right_point: ' + str(right_point))
    # print('four_points: ' + str(four_points))

    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho))) #4점 무작위 이동

    H = cv.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points)) # 4개 대응점 변환 행렬 H ab
    H_inverse = inv(H) #H^ba

    warped_image = cv.warpPerspective(img, H_inverse, size) #H 변환된 이미지 생성

    # print('test_image.shape: ' + str(test_image.shape))
    # print('warped_image.shape: ' + str(warped_image.shape))

    Ip1 = test_image[top_point[1]:BR[1], top_point[0]:BR[0]]    #Ip1 generate(256*256) : numpy 이미지 배열 (Y,X,c) (좌상단:우하단) 
    Ip2 = warped_image[top_point[1]:BR[1], top_point[0]:BR[0]]  # Ip2 generate(256*256)

    training_image = np.dstack((Ip1, Ip2)) # training image생성, Ip1, Ip2 grayscale로 쌓기 2CHANNEL
    # H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
    datum = (training_image, np.array(four_points), np.array(perturbed_four_points))    #특징점 4개 학습(Ip1 Ip2 2channel img + 4points + perturbed 4points)
    
    return datum


### This function is provided by Mez Gebre's repository "deep_homography_estimation"
#   https://github.com/mez/deep_homography_estimation
#   Dataset_Generation_Visualization.ipynb

def process(files, is_test): #input image 전처리 -> .pkl
    if is_test:
        size = (640, 480)
        # Data gen parameters
        rho = 64
        patch_size = 256

    else:
        size = (320, 240)
        # Data gen parameters
        rho = 32
        patch_size = 128

    samples = []
    for f in tqdm(files):
        fullpath = os.path.join(image_folder, f)    # video넣기,, video capture
        img = cv.imread(fullpath, 0)
        img = cv.resize(img, size)
        test_image = img.copy()

        if not is_test:
            for top_point in [(0 + 32, 0 + 32), (128 + 32, 0 + 32), (0 + 32, 48 + 32), (128 + 32, 48 + 32),     #patch를 5곳에서 뽑는다, patch의 좌상단 표시
                              (64 + 32, 24 + 32)]:                          
                # top_point = (rho, rho)
                datum = get_datum(img, test_image, size, rho, top_point, patch_size)
                
                samples.append(datum)
        else:
            top_point = (rho, rho)
            datum = get_datum(img, test_image, size, rho, top_point, patch_size)
            samples.append(datum)

    return samples


if __name__ == "__main__":
    files = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]
    np.random.shuffle(files)

    num_files = len(files)
    print('num_files: ' + str(num_files))

    num_train_files = 100000
    num_valid_files = 8287
    num_test_files = 10000

    train_files = files[:num_train_files]
    valid_files = files[num_train_files:num_train_files + num_valid_files]
    test_files = files[num_train_files + num_valid_files:num_train_files + num_valid_files + num_test_files]

    train = process(train_files, False)
    valid = process(valid_files, False)
    test = process(test_files, True)

    print('num_train: ' + str(len(train)))
    print('num_valid: ' + str(len(valid)))
    print('num_test: ' + str(len(test)))

    with open(train_file, 'wb') as f:
        pickle.dump(train, f)
    with open(valid_file, 'wb') as f:
        pickle.dump(valid, f)
    with open(test_file, 'wb') as f:
        pickle.dump(test, f)
