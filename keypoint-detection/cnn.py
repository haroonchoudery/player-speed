from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from init import *

import tensorflow as tf
from datetime import datetime
from mobilenet_custom import MobileNetCustom
import keras
from keras import backend as K


import keras.applications.mobilenet as mobilenet
from keras.applications.mobilenet import DepthwiseConv2D#, _depthwise_conv_block, _conv_block



def build_model():
	input_shape = (R_HEIGHT, R_WIDTH, CHANNELS)
	#if CHANNELS == 1: input_shape = (WIDTH,HEIGHT)
	model = MobileNetCustom(size=input_shape, classes=NUM_CLASSES).model

	opt = keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
	#opt = keras.optimizers.SGD(lr=LEARNING_RATE, momentum=0.9, nesterov=True)
	#opt = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
	#opt = keras.optimizers.RMSprop(lr=LEARNING_RATE, rho=0.9, epsilon=1e-08, decay=0.0)

	model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

	print("****", model.name, "****")

	return model


def init_model(sess, do_load_model):

	K.set_session(sess)
	K.set_image_dim_ordering('tf')

	model = build_model()

	if do_load_model:
		model_name = get_model_name(model.name)
		model_path = os.path.join(CHECKPOINT_DIR, model_name + ".h5py")
		model.load_weights(model_path) #model = load_model(model.name)

	model.summary()
	return model

def get_model_name(name):
	return name + '_marker_' + str(CHANNELS) + '_' + str(WIDTH)

def save_model(model):
	model_name = get_model_name(model.name)
	model_path = os.path.join(CHECKPOINT_DIR, model_name + ".h5py")
	model.save(model_path)
	print("%s: Model saved in file: %s" % (datetime.now(), model_path))

def load_model(name):
	model_name = get_model_name(name)
	model_path = os.path.join(CHECKPOINT_DIR, model_name + ".h5py")
	print ('loading model:', model_path)
	return keras.models.load_model(model_path,custom_objects=custom_objects)
