import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import uuid

"""
Template Matching
"""

PATCH_L_PATH = 'template_matching/patches/patch-left.jpg'
PATCH_R_PATH = 'template_matching/patches/patch-right.jpg'
CROP = False
DISPLAY = False

def get_match(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = img.shape
    # Load templates
    template_l = cv2.imread(PATCH_L_PATH,0)
    template_r = cv2.imread(PATCH_R_PATH,0)
    w, h = template_r.shape[::-1]

    # All the 6 methods for comparison in a list
    meth = 'cv2.TM_CCOEFF_NORMED'
    method = eval(meth)

    # Apply template Matching
    res_l = cv2.matchTemplate(img,template_l,method)
    res_r = cv2.matchTemplate(img,template_r,method)

    if res_l.max() > res_r.max():
        res = res_l
    else:
        res = res_r

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    x, y = top_left

    out_img = img[y:y+h, x:x+w]
    out_img = cv2.resize(out_img, (w, h))

    return out_img, top_left
