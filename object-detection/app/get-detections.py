import os
import cv2
import numpy as np
import coco
import utils
import model as modellib
import visualize
import pickle

# Mask-R-CNN
MASKRCNN_DIR = 'maskrcnn'
MODEL_DIR = os.path.join("logs")
COCO_MODEL_PATH = os.path.join("mask_rcnn_coco.h5")

VIDEO_DIR = 'videos'
VIDEO_FILE = 'transition.mp4'
video = os.path.join(VIDEO_DIR, VIDEO_FILE)

def to_mot_format(frame_idx, coord):
    """
    Input coordinates:
    (y1, x1, y2, x2)

    Output coordinates:
    (frame, id, bb_left, bb_top, bb_width, bb_height, -1, -1, -1, -1)
    """
    padding = np.array([-1, -1, -1, -1])

    coord = np.insert(coord, 0, frame_idx)
    coord = np.insert(coord, 1, -1)
    coord = np.append(coord, padding)

    width = coord[5] - coord[3]
    height = coord[4] - coord[2]

    coord[4] = width
    coord[5] = height

    # Rearrange coordinates
    coord = coord[[0, 1, 3, 2, 4, 5, 6, 7, 8, 9]]

    return coord

def get_detections_frame(image, frame_idx):
    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    results = model.detect([image], verbose=0)
    rois = results[0]['rois']

    detections = np.zeros([len(rois), 10])

    for idx, coord in enumerate(rois):
        detections[idx] = to_mot_format(frame_idx, coord)

    return detections

def get_detections_video(video):
    """
    Get ROI detections from video using Mask-R-CNN and save in 
    MOTChallenge format
    """    
    camera = cv2.VideoCapture(video)
    num_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    success,image = camera.read()
    count = 0
    success = True
    det_file = open('detections.txt', 'ab')
    
    while success:
        success,image = camera.read()
        print("Processing image {}".format(count))
        detection = get_detections_frame(image, count)
        
        np.savetxt(det_file, detection, delimiter=',', fmt='%1.2f')
        print("Detections extracted from image {}".format(count))

#         if count == 0:
#             detections = detection
#         else:
#             detections = np.concatenate((detections, detection))
            
        count += 1
        det_file.flush()
        
    det_file.close()
    
    print("Finished!")


if __name__ == '__main__':
    detections = get_detections_video(video)

#    with open('detections.pickle', 'wb') as handle:
#        pickle.dump(detections, handle, protocol=pickle.HIGHEST_PROTOCOL)

#    with open('detections.pickle', 'rb') as handle:
#        detections = pickle.load(handle)
