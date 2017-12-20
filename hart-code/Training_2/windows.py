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

def windows(img,win_width,win_height):
    imgWidth = 1024
    imgHeight = 576

    windowWidth = win_width
    windowHeight = win_height

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

    basewidth = 288 #original image base*2
    for i in range(0, numXWin):
        for j in range(0, numYWin):
            cropX = i * winXIncrement
            cropY = j * winYIncrement
            cropped = img[cropX:cropX+windowWidth,cropY:cropY+windowHeight,: ] # crop image from cropX to cropX+windowWidth and from cropY to cropY+windowHeight

            #now we scale the cropped image
            wpercent = (basewidth / float(cropped.size[0]))
             hsize = int((float(cropped.size[1]) * float(wpercent))) #find a height proportional to the width
            scaled_img = cropped.resize((basewidth, hsize), PIL.Image.ANTIALIAS)




