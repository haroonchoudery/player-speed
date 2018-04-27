import numpy as np
from matplotlib import pyplot as plt
import cv2
from init import *

BE_IMG_PATH = 'homography/resources/fullcourt.jpg'
l_court = [[190, 170], [190, 330], [2, 330], [2, 170]]
r_court = [[750, 330], [750, 170], [938, 170], [938, 330]]

def warp_frame(img, predictions, top_left):
    be_img = cv2.imread(BE_IMG_PATH)
    height, width, channels = img.shape

    predictions = np.reshape(predictions, [4, 2])
    predictions = np.array(pred_to_px(predictions, top_left))

    FT_L = predictions[0]
    FT_R = predictions[1]
    BL_R = predictions[2]
    BL_L = predictions[3]

    if FT_L[0] > BL_L[0]:
        be = np.array(l_court)
    else:
        be = np.array(r_court)

    h, status = cv2.findHomography(predictions, be)
    img_out = cv2.warpPerspective(img, h, (be_img.shape[1],be_img.shape[0]))
    print(h.shape)

    return img_out

def pred_to_px(coord, top_left):
    left, top = top_left
    px_coord = []

    for a in coord:
        px_coord.append([a[0] * MODEL_WIDTH + left, a[1] * MODEL_HEIGHT + top])

    return px_coord

def show_warped(img_out):
    cv2.imshow('warped image',img_out)
    cv2.waitKey(0)
