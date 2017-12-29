from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from init import *

import tensorflow as tf

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
import random
import time
from object_detection.utils import dataset_util

import imgaug as ia
from scipy import misc
from imgaug import augmenters as iaa
import random
import plotter
#np.set_printoptions(threshold=np.nan)


imageDim = WIDTH


def create_tf_example(img_path, kp_path):
	img = cv2.imread(img_path, 3)

	df = pd.read_csv(kp_path, header=None, names = ["name", "x", "y"])
	#  df = df.sort_values(['name'])
	rows = df.values

	kp = [];
	for i in range(NUM_POINTS):
		kp.append([])


	height, width, channels = img.shape
	impad = int(width / 2) # int(width / 10)

	index = 0

	for row in rows:
		name = row[0]
		if name != "img":
			xf = float(row[1])
			yf = float(row[2])
			if math.isnan(xf) or math.isnan(yf):
				return False, -1
			x = int(xf * width) + impad
			y = int(yf * height) + impad

			# kp = Keypoints (pixel coordinates)
			kp[index] = [x,y]
			index += 1


	scale = WIDTH/width

	img = cv2.copyMakeBorder(img, impad, impad, impad, impad, cv2.BORDER_CONSTANT, (0,0,0))
	imgHeight, imgWidth, channels = img.shape

	x = []
	y = []

	for coord in kp:
		x.append(coord[0])
		y.append(coord[1])

	minx = min(x)
	maxx = max(x)
	miny = min(y)
	maxy = max(y)

	# Add dynamic padding to fill in remainder of image size (WIDTH, HEIGHT)
	# Can be float (in case dividing by two makes padding a fraction)
	pad_x = (WIDTH - (maxx - minx)) / 2
	pad_y = (HEIGHT - (maxy - miny)) / 2

	left_crop = round(minx - pad_x)
	right_crop = round(maxx + pad_x)
	top_crop = round(miny - pad_y)
	bottom_crop = round(maxy + pad_y)

	# Crop padded image with dynamic padding to fill in remaining pixels from HEIGHT and WIDTH
	crop_img = img[top_crop:bottom_crop, left_crop:right_crop]
	crop_img = cv2.resize(crop_img, (R_WIDTH, R_HEIGHT), interpolation=cv2.INTER_CUBIC)

	# Created adjusted coordinates with cropped images
	kpn = [];
	for i in range(NUM_POINTS):
		kpn.append([])

	for index in range(len(kpn)):
		x = kp[index][0] - left_crop
		y = kp[index][1] - top_crop
		kpn[index] = [x, y]

	joints = []

	scale = R_WIDTH / WIDTH

	# Convert from pixel coordinates to normalized coordinates
	for i, row in enumerate(kpn):
		px, py = row[0], row[1]
		tx = (float(px) * scale)/WIDTH
		ty = (float(py) * scale)/HEIGHT
		joints.append(tx)
		joints.append(ty)

	height  = R_HEIGHT
	width = R_WIDTH

	DEBUG = False

	if DEBUG:
		plot_img = []
		if CHANNELS == 1:
			plot_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
		else:
			plot_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)
		plotter.plot(plot_img,joints)

	crop_img = crop_img.astype(np.uint8)


	channel_transform = cv2.COLOR_BGR2RGB
	if CHANNELS == 1: channel_transform = cv2.COLOR_BGR2YUV
	crop_img = cv2.cvtColor(crop_img, channel_transform)
	if CHANNELS == 1:
		y,u,v = cv2.split(crop_img)
		crop_img = y

	tf_example = tf.train.Example(features=tf.train.Features(feature={
		'image/height': dataset_util.int64_feature(height),
		'image/width': dataset_util.int64_feature(width),
		'image/encoded': dataset_util.bytes_feature(tf.compat.as_bytes(crop_img.tostring())),
		'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
		'image/object/class/label': dataset_util.float_list_feature(joints)
	}))
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
