from init import *
import numpy as np
from matplotlib import pyplot as plt
import cv2
from object_detection.utils import dataset_util

import cnn

def windows(img,win_width,win_height,model):

    height, width, channels = img.shape

    imgWidth = int(width / 2)
    imgHeight = int(height / 2)

    img = cv2.resize(img, (imgWidth, imgHeight), interpolation=cv2.INTER_CUBIC)

    windowWidth = R_WIDTH
    windowHeight = R_HEIGHT

    # number of windows in each direction — total windows = product of these
    numXWin = 10
    numYWin = 5
    total_windows = numXWin*numYWin

    winXIncrement = (imgWidth - windowWidth) / numXWin
    winYIncrement = (imgHeight - windowHeight) / numYWin

    bestCropX = 0
    bestCropY = 0
    bestPredictionScore = 0
    bestPredictions = []

    basewidth = 288 # original image base*2

    predict = -100

    for i in range(0, numXWin):
        for j in range(0, numYWin):
            cropX = int(i * winXIncrement)
            cropY = int(j * winYIncrement)
            cropped = img[cropY:cropY+windowHeight,cropX:cropX+windowWidth,: ] # crop image from cropX to cropX+windowWidth and from cropY to cropY+windowHeight
            predict = model.predict(np.expand_dims(cropped, 0))
            predict_arr = predict[0]
            pred_pix = []

            if np.mean(predict_arr) > 0:
                for index, pred in enumerate(predict_arr):
                    if index % 2 == 0:
                        pred_pix.append((pred * R_WIDTH + cropX) / width)
                    else:
                        pred_pix.append((pred * R_HEIGHT + cropY) / height)

                return pred_pix