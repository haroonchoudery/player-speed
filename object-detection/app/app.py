import cv2
import skvideo.io
from utils import get_detection_bc
import cnn
import tensorflow as tf
import windows

"""
SUDO CODE
"""

def init_model():
    graph = tf.Graph()

    with tf.Session(graph = graph) as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        model = cnn.init_model(sess, False)

    return model
        
def get_keypoints(frame):
    """
    Get keypoints from frame
    """

        
    return kp


def get_detections_bc(frame):
    """
    Iterate through each track in tracker
    Create dictionary with each key = track & value = track.track_id
    
    dict = {'1': {'px_coord' = [], 'real_coord' = []}, 
    '2': {'px_coord' = [], 'real_coord' = []}}
    
    Return bottom center get_detection_bc(tlwh_array)
    """
    
    
    

def warp_frame(frame, kp, detections):
    """
    1. Take frame, use keypoints to warp
    2. Take detections of players and warp onto birdeye view
    """
    
    
    return birdseye
    
    
    
def calc_speed(coord_2, coord_1):
    speed = #do speed calcuation here
    return speed
    
if __name__ == '__main__':
    videodata = skvideo.io.vread(video)
    
    win_width = R_WIDTH
    win_height = R_HEIGHT

    model = init_model()
        
    
    # pass frames through here one by one
    for frame_idx in range(len(videodata)):
        frame = videodata[frame_idx]
        
        # Detect keypoints in frame        
        predict = windows.windows(frame,win_width,win_height,model)     
        kp = get_keypoints(frame)
        
        # Get bottom-center px coordinates of player detections
        detections = get_detections_bc(frame)
        
        # Warp frame using keypoints
        # Return new coordinates of bounding boxes on b/e view
        birdseye = warp_frame(frame, kp, detections)