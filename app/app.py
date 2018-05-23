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
from object_tracking import get_detections
from object_tracking import player_model

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.tracker import Tracker

DEBUG_PREDICTIONS = False
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

def to_input_format(img):
    resized_image = cv2.resize(img, (MODEL_WIDTH, MODEL_HEIGHT))
    model_input = np.expand_dims(resized_image, -1)
    model_input = np.expand_dims(model_input, 0)
    return model_input

def detect_players(frame_no, frame):
    # get all detections from frame
    detections = get_detections.get_detections_frame(frame, frame_no)
    # TODO: run through NN to test which are players and which aren't

    return detections


if __name__ == '__main__':
    # model = load_model()

    for frame_no, frame in enumerate(os.listdir('test_images')):
        """Get keypoint predictions"""
        # Load frame
        frame = cv2.imread(os.path.join('test_images', frame))

        max_cosine_distance = 0.3
        nn_budget = None
        nms_max_overlap = 1.0

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)

        detections = detect_players(frame_no, frame)

        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update >1 :
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)


        cv2.imshow('', frame)








        #
        # # Get black-and-white detection from frame of court
        # court_detection, top_left = template_matching.get_match(frame)
        # # Resize detection to model input size
        # model_input = to_input_format(court_detection)
        # # Feed detection into keypoint detection algorithm and get
        # # pixel coordinates for each point
        # predictions = model.predict(model_input, batch_size = 1)[0]
        #
        # if DEBUG_PREDICTIONS:
        #     labels = [0] * 8
        #     plotter.plot(model_input, labels, predictions)
        #     break
        #
        # warped = homography.warp_frame(frame, predictions, top_left)
        # homography.show_warped(warped)
