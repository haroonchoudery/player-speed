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
import plotter
import PIL
from PIL import Image

def windows(img_path,win_width,win_height,model):

    img = cv2.imread(img_path)
    height, width, channels = img.shape

    imgWidth = width / 2
    imgHeight = height / 2

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

    for i in range(0, numXWin):
        for j in range(0, numYWin):
            cropX = i * winXIncrement
            cropY = j * winYIncrement
            cropped = img[cropX:cropX+windowWidth,cropY:cropY+windowHeight,: ] # crop image from cropX to cropX+windowWidth and from cropY to cropY+windowHeight

            # #now we scale the cropped image
            # wpercent = (basewidth / float(cropped.size[0]))
            #  hsize = int((float(cropped.size[1]) * float(wpercent))) #find a height proportional to the width
            # scaled_img = cropped.resize((basewidth, hsize), PIL.Image.ANTIALIAS)

            predict = model.predict(cropped)

            if np.mean(predict) > 0:
                return predict, cropped

            #find center of window
            #window_center = [(cropX+cropX+windowWidth)/2,(cropY+cropY+windowHeight)/2]

def __init__():
    img_path = 'window_images/frame_001197.jpg'
    win_width = R_WIDTH
    win_height = R_HEIGHT
    model = cnn.init_model(sess, True)
    lbl = [0.82421875,0.6649305555555556,0.705078125,0.8506944444444444,0.2138671875,0.7552083333333334,0.392578125,0.5954861111111112]


    predict, cropped = windows(img_path, win_width,win_height,model)
    
    plotter.plot(cropped, lbl, lgt)
