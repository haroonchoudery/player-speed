import os
import numpy as np
import time

def fcount(path, map = {}):
  count = 0
  for f in os.listdir(path):
    child = os.path.join(path, f)
    if os.path.isdir(child):
      child_count = fcount(child, map)
      count += child_count + 1 # unless include self
  map[path] = count
  return count

if not 'NUM_CLASSES' in globals():
	NUM_DIMS = 2
	NUM_POINTS = 4
	NUM_CLASSES = NUM_DIMS*NUM_POINTS

	WIDTH,HEIGHT = 1024, 576 # Default 224,224
	CHANNELS = 3
	AUGMENT = True
	USE_DEPTH = False
	BATCH_SIZE = 8
	NUM_EPOCHS = 100 # 10000
	LEARNING_RATE = 0.001
	NOT_EXIST = 0
	TRAINING_SPLIT = 0.9
	EVAL_DIR = os.path.join('models','model','eval')
	DROPBOX_LOCATION = 'data'

	NUM_DATA_BATCHES = 1 # fcount(DROPBOX_LOCATION)
	CHECKPOINT_DIR = os.path.join('checkpoint')

	#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
