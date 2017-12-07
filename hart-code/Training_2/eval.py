from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from init import *

import tensorflow as tf
from datetime import datetime

import math
import matplotlib.pyplot as plt

import cnn
import reader


np.set_printoptions(threshold=np.nan)

graph = tf.Graph()


def show_prediction(images, labels, show_labels):
	with tf.Session(graph = graph) as sess:
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		# Initialize the variables (the trained variables and the epoch counter).
		sess.run(init_op)
		model = cnn.init_model(sess, True)

		# Start the queue runners.
		coord = tf.train.Coordinator()
		try:
			threads = []
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))


			step = 0
			while not coord.should_stop():
				for batch_index in range(BATCH_SIZE):
					if show_labels:
						img, lbl = sess.run([images, labels])
					else:
						img = sess.run(images)

					if AUGMENT and show_labels:
						img, lbl = reader.augment(img,lbl)
					# else:
					# 	img = img.astype(np.float32)
					# 	img = img / 255.0

					lgt = model.predict(img, BATCH_SIZE)



					# img = img * 255
					# img = img.astype(np.uint8)



					for j in range(3):
						image = img[j]
						color_map = None
						if CHANNELS == 1:
							image = image.reshape(WIDTH,HEIGHT)
							color_map = 'gray'
						plt.imshow(image, cmap=color_map)
						for i in range(NUM_POINTS):
							plot_colors = ['g','r','b','c']


							if show_labels:
								print("Width:", WIDTH)
								print("Height:", HEIGHT)
								print(lgt[j][i*NUM_DIMS])
								plt.plot(lbl[j][i*NUM_DIMS]*WIDTH, lbl[j][i*NUM_DIMS+1]*HEIGHT, plot_colors[i] +'x')
							plt.plot(lgt[j][i*NUM_DIMS]*WIDTH, lgt[j][i*NUM_DIMS+1]*HEIGHT, plot_colors[i] + 'o')

						# if (show_labels):
						# 	a = lgt[j][0*NUM_DIMS+2]-lgt[j][1*NUM_DIMS+2]
						# 	b = lbl[j][0*NUM_DIMS+2]-lbl[j][1*NUM_DIMS+2]
						# 	print('pred: '+str(a),'act:' +str(b) ,'diff:'+ str(a-b))

						plt.show()

		except Exception as e:  # pylint: disable=broad-except
			coord.request_stop(e)

		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=10)


def single_image_loader():
	filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(os.path.join("test_images","*.jpg")))

	# Read an entire image file which is required since they're JPEGs, if the images
	# are too large they could be split in advance to smaller files or use the Fixed
	# reader to split up the file.
	image_reader = tf.WholeFileReader()

	# Read a whole file from the queue, the first returned value in the tuple is the
	# filename which we are ignoring.
	_, image_file = image_reader.read(filename_queue)

	# Decode the image as a JPEG file, this will turn it into a Tensor which we can
	# then use in training.
	image = tf.image.decode_jpeg(image_file, channels=CHANNELS)

	img_shape = [HEIGHT, WIDTH, CHANNELS]
	if CHANNELS == 1: img_shape.pop()
	image = tf.reshape(image, img_shape)

	 # Generate batch
	images = tf.train.shuffle_batch(
		[image],
		batch_size=BATCH_SIZE,
		num_threads=1,
		capacity=256 + 3 * BATCH_SIZE,
		min_after_dequeue=256)



	labels = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES])
	return images, labels


def evaluate(single_image, show_predictions):

	"""Eval CIFAR-10 for a number of steps."""
	with graph.as_default() as g:
		if single_image:
			images, labels = single_image_loader()
		else:
			images, labels = reader.inputs('test', batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS)

		while True:
			show_prediction(images, labels, not single_image)
			#time.sleep(1)


def main(argv=None):  # pylint: disable=unused-argument
	if tf.gfile.Exists(EVAL_DIR):
		tf.gfile.DeleteRecursively(EVAL_DIR)
	time.sleep(0.1)
	tf.gfile.MakeDirs(EVAL_DIR)
	evaluate(single_image=False, show_predictions=True)

if __name__ == '__main__':
	tf.app.run()
