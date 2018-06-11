import numpy as np
import cv2
import os
from homography import *
import pandas as pd

local_directory = 'homography_demo'
batch = []

for root, dirs, files in os.walk(local_directory):
	for filename in files:
		parts = os.path.splitext(filename)
		ext = parts[1]

		if (ext == '.jpg'):
			base_path = os.path.join(local_directory,parts[0])
			img_path = base_path +'.jpg'
			kp_path = base_path + '.txt'
			if os.path.exists(img_path) and os.path.exists(kp_path):
				batch.append([img_path, kp_path])

batch_length = len(batch)

for frame, val in enumerate(batch):
	img_path = val[0]
	kp_path = val[1]

	img = cv2.imread(img_path)
	height, width, channels = img.shape
	df = pd.read_csv(kp_path, header=None, names = ["name", "x", "y"])
	rows = df.values

	kp = []
	for i in range(4):
		kp.append([])

	index = 0

	for row in rows:
		name = row[0]
		if name != "img":
			xf = float(row[1])
			yf = float(row[2])

			x = int(xf * width)
			y = int(yf * height)

			# kp = Keypoints (pixel coordinates)
			kp[index] = [x,y]
			index += 1

	kp = np.array(kp)
	img_out = homography(img, kp)
	show_warped(img_out)
