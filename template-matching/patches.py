import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import uuid

FRAME_DIR = 'right-frames'
CROP_DIR = 'cropped-images'
CROP = False
DISPLAY = False

for filename in os.listdir(FRAME_DIR):
    img = cv2.imread(os.path.join(FRAME_DIR, filename),0)
    height, width = img.shape
    # Load templates
    template_l = cv2.imread('patch-left.jpg',0)
    template_r = cv2.imread('patch-right.jpg',0)

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
    if CROP:
        img = cv2.imread(os.path.join(FRAME_DIR, filename))
        pad = int(w * h / 10000)
        crop_l = max(0, x - pad)
        crop_r = min(width, x + w + pad)
        crop_t = max(y - pad, 0)
        crop_b = min(height, y + h + pad)
        crop_img = img[crop_t:crop_b, crop_l:crop_r]
        crop_img = cv2.resize(crop_img, (w, h))
        filename = os.path.join(CROP_DIR, str(uuid.uuid4())+'.jpg')
        cv2.imwrite(filename, crop_img)

    if DISPLAY:
        cv2.rectangle(img,top_left, bottom_right, 255, 2)
        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()
