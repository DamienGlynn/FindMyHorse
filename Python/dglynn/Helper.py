import cv2
import numpy as np

class Helper (object):
	''' This class contains helper methods '''

	def pyramid(image, scale=1.5, minSize=(128, 64)):		
		''' Scales down an image by a factor and keep scaling down until the minSize is reached '''
		yield image # yield the original image.
		while True: # Keep looping over the pyramid.
			# Compute the new dimensions of the image and resize it
			w     = int(image.shape[1] / scale)
			image = Helper.resize(image, width=w)
			# If the resized image does not meet the minSize, then stop constructing the pyramid
			if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
				break
			yield image # yield the next image in the pyramid

	def sliding_window(image, stepSize, windowSize):
		''' Slide a window across the image. '''
		for y in range(0, image.shape[0], stepSize):
			for x in range(0, image.shape[1], stepSize):
				if image[y:y + windowSize[1], x:x + windowSize[0]].shape < (128, 64):
					break
				# yield the current window
				yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

	def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
		# Initialize the dimensions of the image to be resized and get the image size.
		dim = None
		(h, w) = image.shape[:2]
		# If both the width and height are None, then return the original image.
		if width is None and height is None:
			return image
		# Check to see if the width is None.
		if width is None:
			# Calculate the ratio of the height and construct the dimensions
			r = height / float(h)
			dim = (int(w * r), height)
		# Otherwise, the height is None
		else:
			# Calculate the ratio of the width and construct the dimensions.
			r = width / float(w)
			dim = (width, int(h * r))
		# Return the resized image.
		return cv2.resize(image, dim, interpolation=inter)

	def setMaxSize(image, max_size):
		''' Set the maximum size of an image. And rescale the smaller side '''
		(h, w) = image.shape[:2]
		if h > w and h != max_size:
			image = Helper.resize(image, height=max_size)
		elif w > h and w != max_size:
			image = Helper.resize(image, width=max_size)
		return Helper.resize(image)

	def BGRtoRGB(image):
		''' OpenCV represents images in BGR order, convert from BGR to RGB. '''
		return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	def non_max_suppression(boxes, overlapThresh):
		# If there are no boxes, return an empty list.
		if len(boxes) == 0:
			return []
		# Initialize the list of picked indexes.
		pick = []
		boxes = np.array( boxes, dtype=object )
		# Get the coordinates of the bounding boxes.
		x1 = boxes[ : ,0]
		y1 = boxes[ : ,1]
		x2 = boxes[ : ,2]
		y2 = boxes[ : ,3]
		# Compute the area of the bounding boxes and sort the bounding
		# boxes by the bottom-right y-coordinate of the bounding box.
		area = (x2 - x1 + 1) * (y2 - y1 + 1)
		idxs = np.argsort(y2)
		# keep looping while some indexes still remain in the indexes list.
		while len(idxs) > 0:
			# Get the last index in the indexes list, add the index value to
			# the list of picked indexes, then initialize the suppression list
			# (i.e. indexes that will be deleted) using the last index.
			last = len(idxs) - 1
			i = idxs[last]
			pick.append(i)
			suppress = [last]
			# Loop over all indexes in the indexes list.
			for pos in range(0, last):
				# Get the current index.
				j = idxs[pos]
				# Find the largest (x, y) coordinates for the start of
				# the bounding box and the smallest (x, y) coordinates
				# for the end of the bounding box.
				xx1 = max(x1[i], x1[j])
				yy1 = max(y1[i], y1[j])
				xx2 = min(x2[i], x2[j])
				yy2 = min(y2[i], y2[j])
				# Compute the width and height of the bounding box
				w = max(0, xx2 - xx1 + 1)
				h = max(0, yy2 - yy1 + 1)
				# Compute the ratio of overlap between the computed
				# bounding box and the bounding box in the area list.
				overlap = float(w * h) / area[j]
				# If there is sufficient overlap, suppress the current bounding box.
				if overlap > overlapThresh:
					suppress.append(pos)
			# Delete all indexes from the index list that are in the suppression list.
			idxs = np.delete(idxs, suppress)
		# Return only the bounding boxes that were picked.
		return boxes[pick]

	def non_max_suppression_fast(boxes, overlapThresh):
		# if there are no boxes, return an empty list
		if len(boxes) == 0:
			return []
	 
		# if the bounding boxes integers, convert them to floats --
		# this is important since we'll be doing a bunch of divisions
		
		#if boxes.dtype.kind == "i":
		# boxes = boxes.astype("float")

		# boxes = np.array( boxes, dtype=float )
	 
		# initialize the list of picked indexes	
		pick = []
	 
		# grab the coordinates of the bounding boxes
		x1 = boxes[:,0]
		y1 = boxes[:,1]
		x2 = boxes[:,2]
		y2 = boxes[:,3]
	 
		# compute the area of the bounding boxes and sort the bounding
		# boxes by the bottom-right y-coordinate of the bounding box
		area = (x2 - x1 + 1) * (y2 - y1 + 1)
		idxs = np.argsort(y2)
	 
		# keep looping while some indexes still remain in the indexes list
		while len(idxs) > 0:
			# grab the last index in the indexes list and add the
			# index value to the list of picked indexes
			last = len(idxs) - 1
			i = idxs[last]
			pick.append(i)
	 
			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = np.maximum(x1[i], x1[idxs[:last]])
			yy1 = np.maximum(y1[i], y1[idxs[:last]])
			xx2 = np.minimum(x2[i], x2[idxs[:last]])
			yy2 = np.minimum(y2[i], y2[idxs[:last]])
	 
			# compute the width and height of the bounding box
			w = np.maximum(0, xx2 - xx1 + 1)
			h = np.maximum(0, yy2 - yy1 + 1)
	 
			# compute the ratio of overlap
			overlap = (w * h) / area[idxs[:last]]
	 
			# delete all indexes from the index list that have
			idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
	 
		# return only the bounding boxes that were picked using the
		# integer data type
		return boxes[pick].astype("int")