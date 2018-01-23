import numpy as np
from matplotlib import pyplot as plt
import cv2

def homography(img, predictions):
    be_img = cv2.imread('homography_images/fullcourt.png')
    height, width, channels = img.shape

    l_court = [[190, 170], [190, 330], [2, 330], [2, 170]]
    r_court = [[750, 330], [750, 170], [938, 170], [938, 330]]

    predictions = np.reshape(predictions, [4, 2])
    predictions = to_px(predictions, width, height)

    FT_L = predictions[0]
    FT_R = predictions[1]
    BL_R = predictions[2]
    BL_L = predictions[3]

    if FT_L[0] > BL_L[0]:
        be = l_court
    else:
        be = r_court

    h, status = cv2.findHomography(predictions, be)
    img_out = cv2.warpPerspective(img, h, (be_img.shape[1],be_img.shape[0]))

    return img_out

def to_px(coord, width, height):
    px_coord = []

    for x in coord:
        px_coord.append([x[0] * width, x[1] * height])

    return px_coord

def show_warped(img_out):
    plt.imshow(img_out)
