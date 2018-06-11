from init import *
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
import random
import time
from object_detection.utils import dataset_util

import imgaug as ia
from scipy import misc
from imgaug import augmenters as iaa
import random
import cnn
import plotter
import PIL
from PIL import Image

def windows(img,model):

    height, width, channels = img.shape

    imgWidth = width # int(width / 2)
    imgHeight = height # int(height / 2)

    # img = cv2.resize(img, (imgWidth, imgHeight), interpolation=cv2.INTER_CUBIC)

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

            print(pred_arr)

            if all(predict_arr) > 0:
                for index, pred in enumerate(predict_arr):
                    if index % 2 == 0:
                        pred_pix.append((pred * R_WIDTH + cropX) / width)
                    else:
                        pred_pix.append((pred * R_HEIGHT + cropY) / height)

                return pred_pix

            # find center of window
            window_center = [(cropX+cropX+windowWidth)/2,(cropY+cropY+windowHeight)/2]

if __name__ == '__main__':
    img_path = 'window_images/frame_001197.jpg'
    img = cv2.imread(img_path)
    win_width = R_WIDTH
    win_height = R_HEIGHT

    lbl = [0.84125,0.57,0.71,0.88, 0.13625,0.7466666666666667,0.35,0.47]

    graph = tf.Graph()

    with tf.Session(graph = graph) as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # Initialize the variables (the trained variables and the epoch counter).
        sess.run(init_op)
        model = cnn.init_model(sess, False)

        predict = windows(img,model)
        plotter.plot(img, lbl, predict)
