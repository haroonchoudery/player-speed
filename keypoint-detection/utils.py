import numpy as np


def window_params(img_width,img_height):
    '''Takes the image dimensions as input and outputs the windowing parameters to be used'''

    win_width = img_width*0.8    #make the window width and height as proportions of the original image
    win_height = img_height*0.7

    return win_width,win_height