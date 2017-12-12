from keras.layers import Input, Reshape, Conv2D, SeparableConv2D, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D
from keras.models import Model, Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import backend as K


class MobileNetCustom():
	classes = 1000
 
	def mobile_block(self, filter_1, filter_2):
		model = self.model
		model.name = "mobilenet_custom"
		model.add(SeparableConv2D(filter_1,kernel_size=(3,3), strides=(1,1),padding='same')) #
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Conv2D(filter_1,kernel_size=(1,1),strides=(1,1), padding='same'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(SeparableConv2D(filter_2, kernel_size=(3,3), strides=(2,2),padding='same'))#
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Conv2D(filter_2 * 2,kernel_size=(1,1),strides=(1,1), padding='same'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))

	def final_conv_block(self):
		model = self.model
		model.add(SeparableConv2D(512,kernel_size=(3,3), strides=(2,2),padding='same'))#
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Conv2D(1024,kernel_size=(1,1),strides=(1,1), padding='same'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(SeparableConv2D(1024,kernel_size=(3,3), strides=(1,1),padding='same'))#
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Conv2D(1024,kernel_size=(1,1),strides=(1,1), padding='same'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))

	def separable_filters(self):
		model = self.model
		for i in range(5):
			model.add(SeparableConv2D(512,kernel_size=(3,3), strides=(1,1),padding='same'))#
			model.add(BatchNormalization())
			model.add(Activation('relu'))
			model.add(Conv2D(512,kernel_size=(1,1),strides=(1,1), padding='same'))
			model.add(BatchNormalization())
			model.add(Activation('relu'))

	def pool_and_classify(self):
		model = self.model
		model.add(AveragePooling2D(pool_size=(7,7),strides=(1,1)))
		model.add(Flatten())
		model.add(Dense(self.classes))
		#model.add(Activation('relu')) #was 'softmax'

	def __init__(self, size=(224,224,3), classes=(1000)):
		self.classes = classes
		self.create(size)

	def create(self, size):
		self.model = Sequential()
		self.model.add(Conv2D(32,kernel_size=(3,3),strides=(2,2), padding='same', input_shape=size))
		self.mobile_block(32,64)
		self.mobile_block(128,128)
		self.mobile_block(256,256)
		self.separable_filters()
		self.final_conv_block()
		self.pool_and_classify()
