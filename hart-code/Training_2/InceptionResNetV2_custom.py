from mobilenet_custom import MobileNetCustom
import keras

from keras.layers import Input, Reshape, Conv2D, SeparableConv2D, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D
from keras.models import Model, Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import backend as K


import keras.applications.mobilenet as mobilenet
from keras.applications.mobilenet import DepthwiseConv2D#, _depthwise_conv_block, _conv_block

def new_model():
    prev_model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights=None, input_tensor=None, input_shape=(256,144,3), pooling=None, classes=1000)
    x = prev_model.output
    #top_layer = Sequential()
    x=AveragePooling2D(pool_size=(2,2), strides=(1, 1))(x)
    x = Flatten()(x)
    x = Dense(1000)(x)
    model = Model(inputs=prev_model.input,outputs=x)
    return model
