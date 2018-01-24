import numpy as np
import cv2
import os
from homography import *
import pandas as pd

src_img = cv2.imread('frame_000018.jpg')
des_img = cv2.imread('fullcourt.jpg')
be = np.array([[190, 170], [190, 330], [2, 330], [2, 170]])
predictions = np.array([[503, 318], [410, 402], [0, 376], [153, 299]])

h, status = cv2.findHomography(predictions, be)
img_out = cv2.warpPerspective(src_img, h, (des_img.shape[1],des_img.shape[0]))

cv2.imshow('warped image',img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
