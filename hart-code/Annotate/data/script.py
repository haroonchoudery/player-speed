import os
import shutil

SOURCE_DIR = 'batch_0'
DEST_DIR = 'batch_1'

def move_files():
	for file in os.listdir('batch_0'):
		if file.endswith('.txt'):
			
			# get name of frame file
			frame = os.path.splitext(file)[0]+'.jpg'

			# move txt and frame files
			shutil.move(os.path.join(SOURCE_DIR, file), os.path.join(DEST_DIR, file))
			shutil.move(os.path.join(SOURCE_DIR, frame), os.path.join(DEST_DIR, frame))


move_files()