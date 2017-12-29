from init import *
import matplotlib.pyplot as plt

plot_colors = ['g','r','b','c']

def plot(image, labels, preds = None):

	color_map = None
	if CHANNELS == 1:
		image = image.reshape(WIDTH,HEIGHT)
		color_map = 'gray'

	plt.imshow(image, cmap=color_map)
	for i in range(NUM_POINTS):
		plt.plot(labels[i*NUM_DIMS]*WIDTH, labels[i*NUM_DIMS+1]*HEIGHT, plot_colors[i]+'o')
		if (preds != None):
			plt.plot(preds[i*NUM_DIMS]*WIDTH, preds[i*NUM_DIMS+1]*HEIGHT, plot_colors[i]+'x')

	plt.show()
