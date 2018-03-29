import numpy as np
import os
import skvideo.io
import matplotlib as mpl
from matplotlib import pyplot as plt
from deep_sort import nn_matching
from deep_sort.deep_sort_app import create_detections
from deep_sort.application_util import preprocessing
from deep_sort.tracker import Tracker

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def tracking(detections_file):      
    """ Track objects"""
    detections_file = np.load(detections_file)
    min_frame_idx = int(detections_file[:, 0].min())
    max_frame_idx = int(detections_file[:, 0].max())
    min_confidence = -1
    min_detection_height = 0
    nms_max_overlap = 0.3
    
    VIDEO_DIR = 'videos'
    VIDEO_FILE = 'transition.mp4'
    video = os.path.join(VIDEO_DIR, VIDEO_FILE)
    videodata = skvideo.io.vread(video)
    
    display = True
    
    N = 50
    cmap = get_cmap(N)
    
    if (display):
        plt.ion()
        fig = plt.figure()

    for frame_idx in range(min_frame_idx, max_frame_idx + 1):
        
        frame = videodata[frame_idx]
        
        print("Processing frame {}".format(frame_idx + 1))

        # Load image and generate detections.
        detections = create_detections(detections_file, frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        max_cosine_distance = 0.2
        nn_budget = 100
        
        metric = nn_matching.NearestNeighborDistanceMetric(
            "euclidean", max_cosine_distance, nn_budget)
        
        tracker = Tracker(metric)
        tracker.predict()
        tracker.update(detections)
        

        # Update visualization.
        if display:
            ax1 = fig.add_subplot(111, aspect='equal')
            ax1.imshow(frame)
            plt.title(' Tracked Targets')

        # Store results.
        for idx, track in enumerate(tracker.tracks):
            print(track.is_confirmed())
#            if not track.is_confirmed() or track.time_since_update > 1:
#                continue
            bbox = track.to_tlwh()
#            results.append([
#                frame_idx+1, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

            if (display):
                ax1.add_patch(mpl.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, lw=2, ec=cmap(idx)))
                ax1.set_adjustable('box-forced')
                plt.text(bbox[0], bbox[1], str(track.track_id))

        if(display):
            fig.canvas.flush_events()
            plt.draw()
            ax1.cla()

            
if __name__ == '__main__':
    tracking('generated_detections.npy')