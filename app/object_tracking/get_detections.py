import os
import skvideo.io
import numpy as np
import deep_sort.coco
import deep_sort.generate_detections
import utils
import model as modellib
import visualize
import pickle
from deep_sort.detection import Detection


# Mask-R-CNN
MASKRCNN_DIR = 'maskrcnn'
MODEL_DIR = os.path.join('resources','logs')
COCO_MODEL_PATH = os.path.join(MODEL_DIR, 'mask_rcnn_coco.h5')

VIDEO_DIR = 'videos'
VIDEO_FILE = 'transition.mp4'
video = os.path.join(VIDEO_DIR, VIDEO_FILE)

# Deep SORT
CHECKPOINT_PATH = os.path.join('resources', 'networks', 'mars-small128.pb')
encoder = deep_sort.generate_detections.create_box_encoder(CHECKPOINT_PATH)
video = os.path.join(VIDEO_DIR, VIDEO_FILE)
name_out = 'generated_detections'
detection_file = 'detections.txt'

class InferenceConfig(deep_sort.coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

def to_mot_format(frame_idx, coord, conf):
    """
    Input coordinates:
    (y1, x1, y2, x2)

    Output coordinates:
    (frame, id, bb_left, bb_top, bb_width, bb_height, -1, -1, -1, -1)
    """
    filler = -1
    bb_left = coord[1]
    bb_top = coord[0]
    bb_width = coord[3] - coord[1]
    bb_height = coord[2] - coord[0]

    # Rearrange coordinates
    coord = np.array([frame_idx + 1,
                      filler,
                      bb_left,
                      bb_top,
                      bb_width,
                      bb_height,
                      conf,
                      filler,
                      filler,
                      filler])

    return coord

def get_detections_frame(image, frame_idx):
    results = model.detect([image], verbose=0)

    rois_all = results[0]['rois']
    confs_all = results[0]['scores']
    class_ids_all = results[0]['class_ids']
    num_detections = len(rois_all)

    # Only use ROIs and confidence levels for people detections
    rois = [rois_all[p] for p in range(num_detections) if class_ids_all[p] == 1]
    confs = [confs_all[p] for p in range(num_detections) if class_ids_all[p] == 1]

    detections = np.zeros([len(rois), 10])

    for idx, coord in enumerate(rois):
        conf = confs[idx]
        detections[idx] = to_mot_format(frame_idx, coord, conf)

    features = encoder(image, detections[:, 2:6].copy())
    final_detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(detections[:,2:6], features)]

    return final_detections

if __name__ == '__main__':
    detections = get_detections_video(video)
