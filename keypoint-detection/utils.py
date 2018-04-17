import numpy as np
from init import *


def window_params(img_width,img_height):
    '''Takes the image dimensions as input and outputs the windowing parameters to be used'''

    win_width = int(img_width * WIN_X_SCALER)    #make the window width and height as proportions of the original image
    win_height = int(img_height * WIN_Y_SCALER)

    return win_width,win_height
