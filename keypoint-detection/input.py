from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from init import *

import tensorflow as tf

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
import random
import time
from object_detection.utils import dataset_util
import imgaug as ia
import plotter
#np.set_printoptions(threshold=np.nan)


def create_tf_example(img_path, kp_path):
	if CHANNELS == 1:
		img_in = cv2.imread(img_path, 0)
		img_in = np.expand_dims(img_in, axis=-1)
	else:
		img_in = cv2.imread(img_path, 3)

	height, width, channels = img_in.shape

	df = pd.read_csv(kp_path, header=None, names = ["name", "x", "y"])
	rows = df.values
	index = 0

	joints = []
	for row in rows:
		name = row[0]
		if name != "img":
			x = float(row[1])
			y = float(row[2])
			if x == 0 and y == 0:
				x = -1
				y = -1
				joints.append(x)
				joints.append(y)
				index += 1
			else:
				joints.append(x)
				joints.append(y)
				index += 1

	img_out = cv2.resize(img_in, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_CUBIC)
	img_out = img_out.astype(np.uint8)

	DEBUG = False

	if DEBUG:
		plot_img = []
		if CHANNELS == 1:
			plot_img = img_out
			# plot_img = cv2.cvtColor(img_out,cv2.COLOR_BGR2GRAY)
		else:
			plot_img = cv2.cvtColor(img_out,cv2.COLOR_BGR2RGB)
		plotter.plot(plot_img,joints)

	tf_example = tf.train.Example(features=tf.train.Features(feature={
	'image/height': dataset_util.int64_feature(height),
	'image/width': dataset_util.int64_feature(width),
	'image/encoded': dataset_util.bytes_feature(tf.compat.as_bytes(img_out.tostring())),
	'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
	'image/object/class/label': dataset_util.float_list_feature(joints)}))

	return True, tf_example


def read_inputs(do_write):
	startIndex = input("What batch do you want to start on?")
	if startIndex == "": startIndex = "0"
	for i in range(int(startIndex),NUM_DATA_BATCHES):
		batch_name = "batch_"+str(i)
		dropped_images = 0
		if do_write:
			record_path = os.path.join('data', str(i)+'_'+ str(CHANNELS) + "_")
			train_writer = tf.python_io.TFRecordWriter(record_path+'train.record')
			val_writer = tf.python_io.TFRecordWriter(record_path+'val.record')
			test_writer = tf.python_io.TFRecordWriter(record_path+'test.record')
			text_file = open(record_path+"length.txt", "w")
		else:
			print('not writing to record')


		local_directory = os.path.join(DROPBOX_LOCATION, 'annotated', batch_name)
		batch = []


		# enumerate local files recursively
		for root, dirs, files in os.walk(local_directory):
			for filename in files:

				parts = os.path.splitext(filename)
				ext = parts[1]

				if (ext == '.jpg'):
					base_path =os.path.join(local_directory,parts[0])
					img_path = base_path +'.jpg'
					kp_path = base_path + '.txt'
					if os.path.exists(img_path) and os.path.exists(kp_path):
						batch.append([img_path, kp_path])


		batch_length = len(batch);
		print (batch_name, batch_length,'images')
		print('')
		for frame, val in enumerate(batch):
			success, tf_example = create_tf_example(val[0], val[1])

			if success:
				percent_done = float(frame) / batch_length
				print('percent=%.3f' % (percent_done), end="\r")
				time.sleep(0.0001)
				if do_write:
					tf_string = tf_example.SerializeToString()
					rnd = random.uniform(0, 1)
					if rnd < 0.99:
						train_writer.write(tf_string)
					#elif rnd >= 0.7 and rnd < 0.85:
					#	val_writer.write(tf_string)
					else:
						test_writer.write(tf_string)
			else:
				dropped_images += 1

		#print('dropped images:', dropped_images)
		total_images = batch_length - dropped_images



		if do_write:
			train_writer.close()
			val_writer.close()
			test_writer.close()
			text_file.write(str(total_images))
			text_file.close()

	#end loop

def main(_):
	read_inputs(True)


if __name__ == '__main__':
	tf.app.run()