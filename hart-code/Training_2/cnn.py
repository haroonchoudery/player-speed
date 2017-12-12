from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from init import *

import tensorflow as tf
import tempfile

from datetime import datetime

# from mobilenet_custom import MobileNetCustom
from InceptionResNetV2_custom import InceptionResNetV2_custom

import keras

from keras.layers import Input, Reshape, Conv2D, SeparableConv2D, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D
from keras.models import Model, Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import backend as K


import keras.applications.mobilenet as mobilenet
from keras.applications.mobilenet import DepthwiseConv2D#, _depthwise_conv_block, _conv_block



def build_model():
	input_shape = (HEIGHT, WIDTH, CHANNELS)
	#if CHANNELS == 1: input_shape = (WIDTH,HEIGHT)
	model = InceptionResNetV2_custom(size=input_shape, classes=NUM_CLASSES).model
	# model = mobile_net(
	# 	input_shape=None,
	# 	alpha=2.0,
	# 	depth_multiplier=1,
	# 	dropout=0,#1e-3,
	# 	include_top=True,
	# 	weights=None,
	# 	input_tensor=None,
	# 	pooling=None,
	# 	classes=NUM_CLASSES)


	# model = Xception(include_top=False, weights=None,
	# 		 input_tensor=None, input_shape=input_shape,
	# 		 pooling='avg',
	# 		 classes=NUM_CLASSES)

	#model = squeezenet.SqueezeNet(classes=NUM_CLASSES)
	#model = darknet.darknet19(input_shape=input_shape, classes=NUM_CLASSES)

	#model = build_sequential()

	#opt = keras.optimizers.Adagrad(lr=LEARNING_RATE, epsilon=1e-08, decay=0.0)
	opt = keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
	#opt = keras.optimizers.SGD(lr=LEARNING_RATE, momentum=0.9, nesterov=True)
	#opt = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
	#opt = keras.optimizers.RMSprop(lr=LEARNING_RATE, rho=0.9, epsilon=1e-08, decay=0.0)

	model.compile(loss='mse',
		optimizer=opt,
		metrics=['accuracy'])

	print("****", model.name, "****")

	return model


def init_model(sess, do_load_model):

	K.set_session(sess)
	K.set_image_dim_ordering('tf')

	model = build_model()

	if do_load_model:
		model_name = get_model_name(model.name)
		model_path = os.path.join(CHECKPOINT_DIR, model_name + ".h5py")
		model.load_weights(model_path)#model = load_model(model.name)

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
	#custom_objects = {'DepthwiseConv2D': mobilenet.DepthwiseConv2D} #'relu6': mobilenet.relu6,
	return keras.models.load_model(model_path,custom_objects=custom_objects)


# def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):

#     channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
#     filters = int(filters * alpha)
#     x = Conv2D(filters, kernel,
#                padding='same',
#                use_bias=False,
#                strides=strides,
#                name='conv1')(inputs)
#     x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
#     return Activation('relu', name='conv1_relu')(x)


# def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
#                           depth_multiplier=1, strides=(1, 1), block_id=1):

#     channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
#     pointwise_conv_filters = int(pointwise_conv_filters * alpha)

#     x = DepthwiseConv2D((3, 3),
#                         padding='same',
#                         depth_multiplier=depth_multiplier,
#                         strides=strides,
#                         use_bias=False,
#                         name='conv_dw_%d' % block_id)(inputs)
#     x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
#     x = Activation('relu', name='conv_dw_%d_relu' % block_id)(x)

#     x = Conv2D(pointwise_conv_filters, (1, 1),
#                padding='same',
#                use_bias=False,
#                strides=(1, 1),
#                name='conv_pw_%d' % block_id)(x)
#     x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
#     return Activation('relu', name='conv_pw_%d_relu' % block_id)(x)

# def mobile_net(input_shape=None,
# 		alpha=1.0,
# 		depth_multiplier=1,
# 		dropout=1e-3,
# 		include_top=True,
# 		weights='imagenet',
# 		input_tensor=None,
# 		pooling=None,
# 		classes=1000):


# 	input_shape = (WIDTH, HEIGHT, CHANNELS)



# 	img_input = Input(shape=input_shape)


# 	x = _conv_block(img_input, 32, alpha, strides=(2, 2))
# 	x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

# 	x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
# 	x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

# 	x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
# 	x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

# 	x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
# 	x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
# 	x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
# 	x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
# 	x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
# 	x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

# 	x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)
# 	x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

# 	shape = (1, 1, int(1024 * alpha))

# 	#Majic sauz
# 	x = GlobalAveragePooling2D()(x)
# 	x = Reshape(shape, name='reshape_1')(x)
# 	x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
# 	x = Reshape((classes,), name='reshape_2')(x)

# 	#x = Dropout(dropout, name='dropout')(x)
# 	# x = AveragePooling2D(pool_size=(7,7),strides=(1,1))(x)
# 	# x = Flatten()(x)
# 	# x = Dense(classes)(x)

# 	# x = GlobalAveragePooling2D()(x)
# 	# x = Reshape(shape, name='reshape_1')(x)
# 	# x = Dropout(dropout, name='dropout')(x)
# 	# x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
# 	#x = Activation('relu', name='act_relu')(x)  #Activation('softmax', name='act_softmax')(x)
# 	#x = Reshape((classes,), name='reshape_2')(x)

# 	# Create model.
# 	model = Model(img_input, x, name='mobilenet')

# 	return model
