import plotter
import cnn
import cv2
import numpy as np
import os
import tensorflow as tf


image_path = '/Users/haroonchoudery/Desktop/frame_000396.jpg'
weights_path = os.path.join('checkpoint', 'model_1_marker_3_800.h5py')
do_load_model = False

labels = [.5,.5,.5,.5,.5,.5,.5,.5]

image = cv2.imread(image_path, 1)

height, width, channels = image.shape

input_image = np.expand_dims(image, axis=0)

with tf.Session() as sess:
    model = cnn.init_model(sess, do_load_model)

model.load_weights(weights_path)

preds = model.predict(input_image)[0]
print(preds)

plotter.plot(cv2.cvtColor(image,cv2.COLOR_BGR2RGB), labels, preds = preds)
