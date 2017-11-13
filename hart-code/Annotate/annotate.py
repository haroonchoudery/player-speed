from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

if hasattr(__builtins__, 'raw_input'):
    input = raw_input

import time
import numpy as np
import pandas as pd
import cv2
import math
import os
from vector import Vector

NOT_EXIST = 0
DATA_LOCATION = "data"

class Annotator:
	directory = ""
	files = []
	lastX = 0
	lastY = 0
	currentFilename = ""
	currentFileIndex = 0
	currentAssignmentId = None
	window_name='Annotate Image'
	window_open=True
	window_scale = 1.0
	data = {}
	impad = 0
	pointFields = ['FT_LEFT','FT_RIGHT','BL_RIGHT','BL_LEFT']
	points = None
	dragPointIndex = -1
	height, width = 0, 0
	img = None
	dirty = False
	waiting = False

	def __init__(self):
		cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
		cv2.setMouseCallback(self.window_name, self.mouseCallback)

	def wait(self):
		self.waiting = True
		keypress = -1
		while self.waiting:
			keypress = cv2.waitKey(1)
			window_closed = cv2.getWindowProperty(self.window_name, 0) == -1
			if window_closed or keypress == 27:
				self.waiting = False
				if self.dirty:
					self.writeCSV()
				self.window_open = False
				cv2.destroyAllWindows()
			else:
				if keypress == ord('r'):
					self.removeKeyPoint()
				elif keypress > -1:
					self.waiting = False
						
		return keypress
	


	def pointExists(self, pt):
		return not (pt[0] == NOT_EXIST and pt[1] == NOT_EXIST)

	def drawMarkers(self):
		cloned = self.img.copy()

		
		pts = []
		plot_colors = [(0,255,0),(0,0,255),(255,255,0),(255,0,0)]
		for i in range(len(self.points)):
			xy = self.points[i]
			x = int(xy[0]*self.width)
			y = int(xy[1]*self.height)
			pt = [x,y]
			
			if self.pointExists(pt):
				pt[0] += self.impad
				pt[1] += self.impad
				pts.append(pt)
				cv2.circle(cloned,(pt[0],pt[1]), 6, plot_colors[i], -1)
				cv2.putText(cloned, self.pointFields[i], (pt[0]+10,pt[1]), cv2.FONT_HERSHEY_TRIPLEX, self.window_scale/2, plot_colors[i])

		if len(pts) > 1:
			for i in range(len(pts)):
				p1 = pts[i] 
				p2 = pts[0]
				if i < len(pts) - 1: 
					p2 = pts[i+1] 
				v1 = Vector(p1[0],p1[1])
				v2 = Vector(p2[0],p2[1])

				diff = (v2-v1)
				unit = diff.normalize()
				distance = diff.norm()
				numPoints = int(distance / 10)
				for i in range(numPoints+1):
					c = v1 + unit * (i * distance / numPoints)
					dp = (int(c[0]),int(c[1]))
					cv2.circle(cloned, dp, 3, (255,0,255), -1)

			

		cv2.imshow(self.window_name,cloned)




	def writeCSV(self):
		csv = ''
		for idx, pointName in enumerate(self.pointFields):
			x = str(self.points[idx][0])
			y = str(self.points[idx][1])
			csv += pointName + ',' + x +','+ y + "\n" #',' + z + 
		text_file = open(self.pathForExt('txt'), "w")
		text_file.write(csv)
		text_file.close()
	

	def readCSV(self):
		self.dirty = False
		image_file = self.files[self.currentFileIndex]
		self.currentFilename = os.path.splitext(image_file)[0]
		filepath = os.path.join(self.directory, self.currentFilename+".txt")

		self.points = []

		
		if os.path.exists(filepath): 
			rows = pd.read_csv(filepath, header=None, names = ["name", "x", "y"]).values
			for row in rows:
				xf = float(row[1])
				yf = float(row[2])
				if math.isnan(xf) or math.isnan(yf):
					print ('NAN detected')
					xf = yf = zf = NOT_EXIST
				self.points.append([xf,yf])
		else:
			self.dirty = True
			for i in range(len(self.pointFields)):
				self.points.append([NOT_EXIST,NOT_EXIST])
			
				

		self.modifyPoints()	

	def modifyPoints(self):
		
		depth_path = self.pathForExt('h5')
		img_path = self.pathForExt('jpg')
		img_data = cv2.imread(img_path)
		h,w,c = img_data.shape 

		
		self.img = cv2.resize(img_data, (int(w * self.window_scale), int(h * self.window_scale)), interpolation = cv2.INTER_CUBIC)
		

		self.height, self.width, channels = self.img.shape
		pad = int((self.height*self.width) / 2500)
		self.impad = pad
		self.img = cv2.copyMakeBorder( self.img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, (0,0,0))

		self.drawMarkers()
		keypress = self.wait()
		if keypress == ord('f'):
			self.nextPrev(-1)
		elif keypress == ord('d'):
			self.nextPrev(1)
		elif keypress == ord('s'):
			self.dirty = False
			self.nextPrev(-1)
	
	def closestPointIndex(self,x,y):
		x -= self.impad
		y -= self.impad
		closestDist = 50
		closestIndex = -1
		for idx, point in enumerate(self.points):
			dist = math.hypot(x - point[0]*self.width, y - point[1]*self.height)
			if dist < closestDist:
				closestDist = dist
				closestIndex = idx

		return closestIndex

	def nextPrev(self, flags):
		if flags < 0:
			if self.currentFileIndex < len(self.files) - 1:
				self.currentFileIndex += 1  
		else:
			if self.currentFileIndex > 0: 
				self.currentFileIndex -= 1
		print (self.currentFileIndex)
		if self.dirty: 
			self.writeCSV()
	

	def mouseMoved(self, x, y):
		if self.dragPointIndex == -1:
			return
		x -= self.impad
		y -= self.impad
		pt = [x/self.width, y/self.height]
		self.points[self.dragPointIndex] = pt
		self.dirty = True
		self.drawMarkers()

	def removeKeyPoint(self):
		removalIndex = self.closestPointIndex(self.lastX,self.lastY)
		if removalIndex == -1:
			if self.pointExists(self.points[1]):
				removalIndex = 1
			else:
				removalIndex = 0
		self.points[removalIndex] = [NOT_EXIST,NOT_EXIST]
		self.dirty = True
		self.drawMarkers()

	def mouseCallback(self,event,x,y,flags,param):

		if event == cv2.EVENT_LBUTTONDOWN:
			self.dragPointIndex = self.closestPointIndex(x,y)
			if self.dragPointIndex == -1:
				for i in range(len(self.pointFields)):
					if not self.pointExists(self.points[i]):
						self.dragPointIndex = i
						self.mouseMoved(x,y)
						break;

		elif event == cv2.EVENT_MOUSEMOVE:
			self.lastX, self.lastY = x, y
			self.mouseMoved(x,y)
		elif event == cv2.EVENT_LBUTTONUP:
			self.dragPointIndex = -1

	def pathForExt(self,ext):
		return os.path.join(self.directory, self.currentFilename + '.' + ext)

	
		
		
def parseVideo(local_directory, filename, rotate):
	vidcap = cv2.VideoCapture(os.path.join(local_directory,filename))
	success,image = vidcap.read()
	count = 0
	frame = 0
	success = True
	spacer = 4
	while success:
		try:
			success,image = vidcap.read()
			if count % spacer == 0:
				if rotate:
					image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE);
				print('Save frame: ', frame)
				height,width,channels = image.shape
				cv2.imwrite(os.path.join(local_directory, "frame_"+str(frame).zfill(5)+".jpg"), image)     # save frame as JPEG file
				frame += 1
			count += 1
		except:
			return


if __name__ == '__main__':
	local_directory = input('Which batch_# directory would you like to edit: ')
	if local_directory == '':
		local_directory = '0'
	local_directory = "batch_" + local_directory
	local_directory = os.path.join(DATA_LOCATION, local_directory)
	# print (local_directory)

	# doParse = input('Would you like to parse video? (y)')

	# if doParse == 'y':
	# 	parseVideo(local_directory, 'data.mp4', False)

	filenames = []
	for root, dirs, files in os.walk(local_directory):
		for filename in files:
			
			if '.jpg' in filename:
				filenames.append(filename)
				
	filenames = sorted(filenames)
	editor = Annotator()
	editor.files = filenames
	editor.directory = local_directory
	while (editor.window_open):
		editor.readCSV()


