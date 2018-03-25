import os

from deep_sort.deep_sort import generate_detections
from deep_sort.deep_sort import Detection

DEEP_SORT_DIR = 'deep_sort'
DETECTION_DIR = 'detections'

VIDEO_DIR = 'videos'
VIDEO_FILE = 'transition.mp4'

CHECKPOINT_PATH = os.path.join(DEEP_SORT_DIR, 'checkpoint', 'mars-small128.ckpt-68577')

encoder = generate_detections.create_box_encoder(CHECKPOINT_PATH)
video = os.path.join(VIDEO_DIR, VIDEO_FILE)
name_out = 'generated_detections.txt'
detection_file = 'detections.txt'

def generate_detections_features(encoder, video, name_out, detection_file):

    """Generate detections with features. Modification with video"""
    detections_in = np.loadtxt(detection_file, delimiter=',')
    detections_out = []

    frame_indices = detections_in[:, 0].astype(np.int)
    min_frame_idx = frame_indices.astype(np.int).min()
    max_frame_idx = frame_indices.astype(np.int).max()

    camera = cv2.VideoCapture(video)

    for frame_idx in range(min_frame_idx, max_frame_idx + 1):
        print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
        mask = frame_indices == frame_idx
        rows = detections_in[mask]

        if frame_idx not in frame_indices:
            print("WARNING could not find image for frame %d" % frame_idx)
            continue
        camera.set(cv2.CAP_PROP_POS_FRAMES, frame_idx-1);
        (grabbed, bgr_image) = camera.read()
#         bgr_image = cv2.imread(image_filenames[frame_idx], cv2.IMREAD_COLOR)
        features = encoder(bgr_image, rows[:, 2:6].copy())
        detections_out += [np.r_[(row, feature)] for row, feature
                           in zip(rows, features)]

    # output_filename = os.path.join(output_dir, "%s.npy" % sequence)
    np.save(name_out, np.asarray(detections_out), allow_pickle=False)
    
generate_detections_features(encoder, video, name_out, detection_file)