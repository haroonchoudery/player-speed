import plotter
import cnn
import cv2
import numpy as np
import os
import tensorflow as tf


image_path = '/Users/haroonchoudery/Desktop/frame_000020.jpg'
weights_path = os.path.join('checkpoint', 'mobilenet_custom_marker_3_1024.h5py')
do_load_model = False

labels = [0.5185546875,0.546875,0.431640625,0.6927083333333334,0.0244140625,0.6440972222222222,0.1787109375,0.5104166666666666]


image = cv2.imread(image_path, 1)

height, width, channels = image.shape
print(height)

reshaped_image = image.reshape(width, height, 3)
input_image = np.expand_dims(reshaped_image, axis=0)

with tf.Session() as sess:
    model = cnn.init_model(sess, do_load_model)

model.load_weights(weights_path)

preds = model.predict(input_image)[0]
print(preds)

plotter.plot(cv2.cvtColor(reshaped_image,cv2.COLOR_BGR2RGB), labels, preds = preds)
