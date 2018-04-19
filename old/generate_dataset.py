from init import *
from utils import *
import cv2
import os
import uuid

def generate_dataset(img_path, dst_dir):
    """
    Takes input directory of raw images and creates a dataset of new images of
    shape (WINDOW_HEIGHT, WINDOW_WIDTH, 3) to be annotated
    """
    img = cv2.imread(img_path)
    height, width, channels = img.shape

    win_x_increment = (width - WINDOW_WIDTH) / NUM_WIN_X
    win_y_increment = (height - WINDOW_HEIGHT) / NUM_WIN_Y

    for i in range(0, NUM_WIN_X):
        for j in range(0, NUM_WIN_Y):
            crop_x = int(i * win_x_increment)
            crop_y = int(j * win_y_increment)
            cropped = img[crop_y:crop_y+WINDOW_HEIGHT,crop_x:crop_x+WINDOW_WIDTH,:]

            # Give file a random name and then save it to dest folder as JPG
            filename = os.path.join(dst_dir, str(uuid.uuid4())+'.jpg')
            cv2.imwrite(filename, cropped)



if __name__ == '__main__':
    src_dir = 'raw_frames'
    dst_dir = 'generated_images'

    for img in os.listdir(src_dir):
        img_path = os.path.join(src_dir, img)
        generate_dataset(img_path, dst_dir)
