from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from init import *

import tensorflow as tf


import imgaug as ia
from imgaug import augmenters as iaa
from scipy import misc
import random
import plotter

np.set_printoptions(threshold=np.nan)

def get_num_files(i):
	len_path = os.path.join('data', str(i) + "_" + str(CHANNELS) + '_length.txt')
	if os.path.exists(len_path):
		with open(len_path, 'r') as myfile:
			return int(myfile.read().replace('\n', ''))
	return 0

def get_total_images():
	total_images = 0
	for i in range(0,NUM_DATA_BATCHES):
		length = get_num_files(i)
		total_images += length

	return total_images


seq = iaa.Sequential([
	iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
	iaa.Fliplr(0.5), # horizontally flip 50% of the images
	#iaa.PerspectiveTransform(scale=(0.01, 0.075)),
	#iaa.GaussianBlur(sigma=(0, 2.0)), # blur images with a sigma of 0 to 2.0
	#iaa.Add((-10, 10)),#, per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
	iaa.Multiply((0.25, 1.5)), #, per_channel=0.5
	iaa.Affine(rotate=(-10, 10), scale=(0.8, 1.00)) # rotate by -10 to +10 degrees, scale up to 80%)
])

def augment(images, labels):
	keypoints_on_images = []

	for label in labels:
		keypoints = []
		for i in range(NUM_POINTS):
			keypoints.append(ia.Keypoint(x=label[i*NUM_DIMS]*WIDTH, y=label[i*NUM_DIMS+1]*HEIGHT))
		keypoints_on_images.append(ia.KeypointsOnImage(keypoints, shape=(WIDTH,HEIGHT,3)))

	seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start
	images_aug = seq_det.augment_images(images)
	keypoints_aug = seq_det.augment_keypoints(keypoints_on_images)

	for idx1, keypoints_after in enumerate(keypoints_aug):

		for idx2, keypoint in enumerate(keypoints_after.keypoints):

			if not (labels[idx1, idx2*NUM_DIMS] == NOT_EXIST and labels[idx1, idx2*NUM_DIMS+1] == NOT_EXIST):
				labels[idx1, idx2*NUM_DIMS] = float(keypoint.x / WIDTH)
				labels[idx1, idx2*NUM_DIMS+1] = float(keypoint.y / HEIGHT)

	# images_aug = images_aug.astype(np.float32)
	# images_aug = images_aug / 255.0

	#print(joints_aug)

	return images_aug, labels

def read_and_decode(filename_queue):
	feature = {'image/encoded': tf.FixedLenFeature([], tf.string),
	'image/object/class/label': tf.FixedLenFeature([NUM_CLASSES], tf.float32)}
	# Create a list of filenames and pass it to a queue

	# Define a reader and read the next record
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	# Decode the record read by the reader
	features = tf.parse_single_example(serialized_example, features=feature)
	# Convert the image data from string back to the numbers
	image = tf.decode_raw(features['image/encoded'], tf.uint8)

	#image = tf.to_float(image, name='ToFloat')
	#image = image / 255.0

	# Reshape image data into the original shape
	img_shape = [WIDTH, HEIGHT, CHANNELS]
	#if CHANNELS == 1: img_shape.pop()
	image = tf.reshape(image, img_shape)


	print('Build model with input:', image.shape)

	label = features['image/object/class/label']


	# #preprocess
	# if training:
	# 	#np_img = image.eval()
	# 	print(image)
	# 	#image = tf.image.random_flip_left_right(image)
	# 	#image = tf.image.random_brightness(image, max_delta=63)
	# 	#image = tf.image.random_contrast(image, lower=0.2, upper=1.8)


	return image, label

def inputs(record_name, batch_size, num_epochs):

	training = record_name == 'train'
	filenames = []

	for i in range(0,NUM_DATA_BATCHES):
		filename = os.path.join('data',str(i) + '_'+ str(CHANNELS) + '_'  + record_name + '.record')
		length = get_num_files(i)
		if length > batch_size:
			filenames.append(filename)

	print ('Loading:', filenames)
	with tf.name_scope('input'):
		filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
		image, label = read_and_decode(filename_queue)
		images, labels = tf.train.shuffle_batch(
			[image, label], batch_size=batch_size, num_threads=2,
			capacity=1000 + 3 * batch_size,
			min_after_dequeue=1000) # Ensures a minimum amount of shuffling of examples.

		return images, labels


def main(_):
	with tf.Session() as sess:

		images, labels = inputs('train', BATCH_SIZE, 10)
		# Initialize all global and local variable
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init_op)
		# Create a coordinator and run all QueueRunner objects
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		for batch_index in range(10):
			img, lbl = sess.run([images,labels])

			images_aug, labels_aug = augment(img, lbl)

			# images_aug = images_aug * 255
			# images_aug = images_aug.astype(np.uint8)

			for j in range(BATCH_SIZE):
				plotter.plot(images_aug[j],labels_aug[j])

		coord.join(threads)
		sess.close()

if __name__ == '__main__':
	tf.app.run()
