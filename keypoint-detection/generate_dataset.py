import init
import utils
import cv2
import os
import uuid

def generate_dataset(img, dst_dir):
    """
    Takes input directory of raw images and creates a dataset of new images of
    shape (window_height, window_width, 3) to be annotated
    """
    height, width, channels = img.shape
    window_width, window_height = window_params(width, height)

    win_x_increment = (imgWidth - windowWidth) / NUM_WIN_X
    win_y_increment = (imgHeight - windowHeight) / NUM_WIN_Y

    for i in range(0, NUM_WIN_X):
        for j in range(0, NUM_WIN_Y):
            crop_x = int(i * win_x_increment)
            crop_y = int(j * win_y_increment)
            cropped = img[crop_y:crop_y+window_height,crop_x:crop_x+window_width,:]

            # Give file a random name and then save it to dest folder as JPG
            filename = os.path.join(dst_dir, str(uuid.uuid4())+'.jpg')
            cv2.imsave(filename, cropped)



if __name__ == '__main__':
    src_dir = 'raw_frames'
    dst_dir = 'generate_images'

    for img in os.listdir(src_dir):
        generate_dataset(img, dst_dir)
