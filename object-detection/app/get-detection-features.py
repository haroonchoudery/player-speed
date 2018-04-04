import os
import deep_sort.generate_detections
import deep_sort.detection
import numpy as np
import skvideo.io

DETECTION_DIR = 'detections'

VIDEO_DIR = 'videos'
VIDEO_FILE = 'transition.mp4'

CHECKPOINT_PATH = os.path.join('resources', 'networks', 'mars-small128.pb')

encoder = generate_detections.create_box_encoder(CHECKPOINT_PATH)
video = os.path.join(VIDEO_DIR, VIDEO_FILE)
name_out = 'generated_detections'
detection_file = 'detections.txt'

def generate_detections_features(encoder, video, name_out, detection_file):

    """Generate detections with features. Modification with video"""
    detections_in = np.loadtxt(detection_file, delimiter=',')
    detections_out = []

    frame_indices = detections_in[:, 0].astype(np.int)
    min_frame_idx = frame_indices.astype(np.int).min()
    max_frame_idx = frame_indices.astype(np.int).max()

    videodata = skvideo.io.vread(video)

    for frame_idx in range(min_frame_idx, max_frame_idx + 1):
        print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
        mask = frame_indices == frame_idx
        rows = detections_in[mask]
       
        image = videodata[frame_idx - 1]
        features = encoder(image, rows[:, 2:6].copy())
        detections_out += [np.r_[(row, feature)] for row, feature
                           in zip(rows, features)]

    np.save(name_out, np.asarray(detections_out), allow_pickle=False)
    
generate_detections_features(encoder, video, name_out, detection_file)