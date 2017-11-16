import plotter
import cnn
import cv2
import numpy as np
import os
import tensorflow as tf


image_path = '/Users/haroonchoudery/Desktop/frame_000228.jpg'
weights_path = os.path.join('checkpoint', 'mobilenet_custom_marker_3_1024.h5py')
do_load_model = False

labels = [0.3935546875,0.8524305555555556,0.24609375,0.6944444444444444,0.642578125,0.6197916666666666,0.83984375,0.7534722222222222]


image = cv2.imread(image_path, 1)

height, width, channels = image.shape

input_image = np.expand_dims(image, axis=0)

with tf.Session() as sess:
    model = cnn.init_model(sess, do_load_model)

model.load_weights(weights_path)

preds = model.predict(input_image)[0]
print(preds)

plotter.plot(cv2.cvtColor(image,cv2.COLOR_BGR2RGB), labels, preds = preds)
