from init import *
import matplotlib.pyplot as plt
import numpy as np

plot_colors = ['g','r','b','c']

def plot(image, labels, preds = None):
	color_map = None
	if CHANNELS == 1:
		image = np.squeeze(image, axis = -1) # Remove last axis
		image = np.squeeze(image, axis = 0)
		color_map = 'gray'

	print(image.shape)
	plt.imshow(image, cmap=color_map)

	for i in range(NUM_POINTS):
		plt.plot(labels[i*NUM_DIMS]*MODEL_WIDTH, labels[i*NUM_DIMS+1]*MODEL_HEIGHT, plot_colors[i]+'o')
		if (preds.all() != None):
			plt.plot(preds[i*NUM_DIMS]*MODEL_WIDTH, preds[i*NUM_DIMS+1]*MODEL_HEIGHT, plot_colors[i]+'x')

	plt.show()
