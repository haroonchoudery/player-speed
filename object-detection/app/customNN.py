from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras import layers
from tensorflow.python.keras._impl.keras.applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras._impl.keras.applications.imagenet_utils import decode_predictions  # pylint: disable=unused-import
from tensorflow.python.keras._impl.keras.engine.topology import get_source_inputs
from tensorflow.python.keras._impl.keras.layers import Activation
from tensorflow.python.keras._impl.keras.layers import BatchNormalization
from tensorflow.python.keras._impl.keras.layers import Conv2D
from tensorflow.python.keras._impl.keras.layers import Dense
from tensorflow.python.keras._impl.keras.layers import Dropout
from tensorflow.python.keras._impl.keras.layers import Flatten
from tensorflow.python.keras._impl.keras.layers import AveragePooling2D
from tensorflow.python.keras._impl.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras._impl.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras._impl.keras.layers import Input
from tensorflow.python.keras._impl.keras.layers import MaxPooling2D
from tensorflow.python.keras._impl.keras.layers import SeparableConv2D
from tensorflow.python.keras._impl.keras.models import Model
from tensorflow.python.keras._impl.keras.utils.data_utils import get_file
from tensorflow.python.platform import tf_logging as logging

class customNN():
    classes = 2

    def mobile_block(self, filter_1, filter_2):
        model = self.model
        model.name = "mobilenet_custom"
        model.add(SeparableConv2D(filter_1, kernel_size=(3, 3), strides=(1, 1), padding='same'))  #
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filter_1, kernel_size=(1, 1), strides=(1, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(SeparableConv2D(filter_2, kernel_size=(3, 3), strides=(2, 2), padding='same'))  #
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filter_2 * 2, kernel_size=(1, 1), strides=(1, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    def final_conv_block(self):
        model = self.model
        model.add(SeparableConv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same'))  #
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(1024, kernel_size=(1, 1), strides=(1, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(SeparableConv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding='same'))  #
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(1024, kernel_size=(1, 1), strides=(1, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    def separable_filters(self):
        model = self.model
        for i in range(5):
            model.add(SeparableConv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same'))  #
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))

    def pool_and_classify(self):
        model = self.model
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1)))
        model.add(Flatten())
        model.add(Dense(self.classes))

    def __init__(self, size=(224, 224, 3), classes=(2)):
        self.classes = classes
        self.create(size)

    def create(self, size):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', input_shape=size))
        self.mobile_block(32, 64)
        self.mobile_block(128, 128)
        self.mobile_block(256, 256)
        self.separable_filters()
        self.final_conv_block()
        self.pool_and_classify()