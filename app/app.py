import cv2
import os
import numpy as np
# import skvideo.io
# from utils import get_detection_bc

import tensorflow as tf

from init import *
import plotter

from keypoint_detection import cnn
from template_matching import template_matching
from homography import homography

DEBUG = True
"""
Steps for each frame:
    KEYPOINT DETECTION
    1. Use template matching to identify area where key is located
    2. Run keypoint detection and return keypoints on keypoint
    3. Use function to get pixel coordinates of keypoints on entire image

    PLAYER DETECTION
    1. Identify people in frame
    2. Run people detections through NN to identify which are players
    3. Return bottom-center coordinates of each bounding box

    PLAYER TRACKING
    1. Get Deep SORT features for each player detected
    2. Pass features to tracker and identify which player belongs to which ID

    HOMOGRAPHY
    1. Use keypoint coordinates and pixel coordiates on birdseye view of court
        to warp image
    2. Get Perspective Transform coordinates of each bottom-center coordinate
        for each bounding box after warp
    3. Convert PT coordinates to real-court coordinates of each player


    For each tracked player, record their location on the court over during any
    given frame


"""

def load_model():
    model = cnn.init_model(do_load_model = True, verbose = False)
    return model

if __name__ == '__main__':
    model = load_model()

    for frame in os.listdir('test_images'):
        # Load frame and black-and-white version of frame
        frame = cv2.imread(os.path.join('test_images', frame))
        # Get detection from frame of court
        court_detection, top_left = template_matching.get_match(frame)
        print(court_detection.shape)
        # Resize detection to model input size
        resized_image = cv2.resize(court_detection, (MODEL_WIDTH, MODEL_HEIGHT))
        model_input = np.expand_dims(resized_image, -1)
        model_input = np.expand_dims(model_input, 0)
        # Feed detection into keypoint detection algorithm and get
        # pixel coordinates for each point
        predictions = model.predict(model_input, batch_size = 1)[0]

        if DEBUG:
            labels = [0] * 8
            plotter.plot(model_input, labels, predictions)
            break

        warped = homography.warp_frame(frame, predictions, top_left)

        homography.show_warped(warped)
