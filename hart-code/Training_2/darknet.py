"""Darknet19 Model Defined in Keras."""
import functools
from functools import partial
from keras import backend as K

from keras.layers import Input, Conv2D, MaxPooling2D, Convolution2D, GlobalAveragePooling2D, Activation, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Flatten, Dense
from keras.models import Model
from keras.regularizers import l2
from functools import reduce

#from ..utils import compose

# Partial wrapper for Convolution2D with static default argument.
_DarknetConv2D = partial(Conv2D, padding='same')

def compose(*funcs):
	"""Compose arbitrarily many functions, evaluated left to right.
	Reference: https://mathieularose.com/function-composition-in-python/
	"""
	# return lambda x: reduce(lambda v, f: f(v), funcs, xok,)
	if funcs:
		return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
	else:
		raise ValueError('Composition of empty sequence not supported.')


@functools.wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
	"""Wrapper to set Darknet weight regularizer for Convolution2D."""
	darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
	darknet_conv_kwargs.update(kwargs)
	return _DarknetConv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
	"""Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
	no_bias_kwargs = {'use_bias': False}
	no_bias_kwargs.update(kwargs)
	return compose(
		DarknetConv2D(*args, **no_bias_kwargs),
		BatchNormalization(),
		LeakyReLU(alpha=0.1))


def bottleneck_block(outer_filters, bottleneck_filters):
	"""Bottleneck block of 3x3, 1x1, 3x3 convolutions."""
	return compose(
		DarknetConv2D_BN_Leaky(outer_filters, (3, 3)),
		DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
		DarknetConv2D_BN_Leaky(outer_filters, (3, 3)))


def bottleneck_x2_block(outer_filters, bottleneck_filters):
	"""Bottleneck block of 3x3, 1x1, 3x3, 1x1, 3x3 convolutions."""
	return compose(
		bottleneck_block(outer_filters, bottleneck_filters),
		DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
		DarknetConv2D_BN_Leaky(outer_filters, (3, 3)))


def darknet_body():
	"""Generate first 18 conv layers of Darknet-19."""
	return compose(
		DarknetConv2D_BN_Leaky(32, (3, 3)),
		MaxPooling2D(),
		DarknetConv2D_BN_Leaky(64, (3, 3)),
		MaxPooling2D(),
		bottleneck_block(128, 64),
		MaxPooling2D(),
		bottleneck_block(256, 128),
		MaxPooling2D(),
		bottleneck_x2_block(512, 256),
		MaxPooling2D(),
		bottleneck_x2_block(1024, 512))


def darknet19(input_shape=None, classes=1000):
	"""Generate Darknet-19 model for Imagenet classification."""
	inputs = Input(shape=input_shape)

	#shape = (1, 1, 1024)
	#dropout=1e-3
	body = darknet_body()(inputs)
	body = DarknetConv2D(classes, (1, 1))(body)#, activation='relu')(body)
	body = Flatten()(body)
	logits = Dense(classes)(body)
	#logits = Reshape((classes,), name='reshape_2')(body)
	#x = Activation('relu', name='act_relu')(x)  #Activation('softmax', name='act_softmax')(x)
	#
	#logits = GlobalAveragePooling2D()(x)


	return Model(inputs, logits, name='darknet19')
