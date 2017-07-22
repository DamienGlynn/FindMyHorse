import cv2
import time
import numpy as np
import skimage.io as io
from dglynn.Helper             import Helper
from dglynn.Hog                import Hog
from dglynn.HorseFeatureVector import HorseFeatureVector
from dglynn.HorseClassifier    import HorseClassifier

data_path = 'datasets/data_new.h5'
imge_path = 'images/test_images_large/1.jpg'

if __name__ == "__main__":
	
	# Load the data from file.
	datasets = HorseFeatureVector.load(data_path)
	
	# Split the data into X samples and y labels.
	X, y = datasets[0], datasets[1]

	# Initialize the Horse classifier.
	horseClassifier = HorseClassifier()

	# Train the classifier.
	horseClassifier.train (X, y)
	
	# Load the image.
	image = io.imread(imge_path, flatten=False)

	# Set the max size of the image.
	image = Helper.setMaxSize(image, 400)

	# Define the window width and height.
	(winW, winH) = (64, 128)
	
	# To store the positive predictions.
	found = []
	
	# Loop over the image pyramid.
	for resized in Helper.pyramid(image, scale=1.5):
		
		# Convert the cv2 BRG image to RGB.
		resized = Helper.BGRtoRGB(resized)

		# Stores the predictions for the current image scale.
		predictions = []

		# Loop over the sliding window for each layer of the pyramid.
		for (x, y, window) in Helper.sliding_window(resized, stepSize=4, windowSize=(winW, winH)):
			
			# If the window does not meet our desired window size, ignore it.
			if window.shape[0] != winH or window.shape[1] != winW:
				continue
			
			# Convert to a 2d array.
			window = window[:,:,-1]
			
			# Calculate the hog descriptor for the current window.
			horse_descriptor = Hog.describe(window)
			
			# Convert to a 2d array.
			horse_descriptor = horse_descriptor.reshape(1, -1)
			
			# Make the predition.
			# predicted = horseClassifier.predict (horse_descriptor)
			
			# Get the decision made by the classifier.
			decision = horseClassifier.decision(horse_descriptor)

			# If there is a match.
			if decision > 2:
				
				# Set the box colour to green.
				box_color = (0, 255, 0)

				# The found box and the current image size.
				box = [x, y, x+winW, y+winH, resized.shape]
				
				# print( 'box: ', box )
				# Add the predicted window to the predictions list.
				predictions.append( box )
				
				print( decision )
			
			else:

				# Set the box colour to red.
				box_color = (0, 0, 255)

			# Display the process.
			clone = resized.copy()

			# Draw each prediction.
			for rect in predictions:
				cv2.rectangle(clone, (rect[0], rect[1]), (rect[0] + winW, rect[1] + winH), (0, 255, 0) )
			cv2.rectangle(clone, (x, y), (x + winW, y + winH), box_color, 1)
			cv2.imshow("Window", clone)
			cv2.waitKey(1)
			time.sleep(0.025)
		
		# Get the best pick of the predictions.
		pick = Helper.non_max_suppression(predictions, 0.2)
		
		# pick = Helper.non_max_suppression_fast(predictions, 0.2)

		# Add the best pick to the found list.
		found.extend(pick)

		# print( 'Predictions:', len(predictions) )
		# print( 'Best Picks :', len(pick) )
		# for p in found:
		# 	print( 'found: ', p )

		# Display the best predictions at the current image scale.
		for (x, y, w, h, size) in pick:
			cv2.rectangle(resized, (x, y), (w, h), (0, 255, 0) )
		cv2.imshow("After NMS", resized)
		cv2.waitKey(0)

	# Show all the best predictions on the original image.
	for rect in found:

		# print( (rect[0], rect[1]), (rect[2], rect[3]) )

		cv2.rectangle(image, (rect[0], rect[1]), ( (rect[0] + rect[2] ), ( rect[1] + rect[3]) ), (0, 255, 0) )
	
	image = Helper.BGRtoRGB(image) # Convert the cv2 BGR image to RGB
	cv2.imshow("Window", image)
	cv2.waitKey(0)

	print( 'Found:', len(found) )

	# # Get the bet pick from the overlapping boxes.
	# pick = Helper.non_max_suppression(found, 0.3)
	# for rect in pick:
	# 	cv2.rectangle(image, (rect[0], rect[1]), ( (rect[0] + rect[2] ), ( rect[1] + rect[3]) ), (0, 255, 0) )
	# cv2.waitKey(0)

