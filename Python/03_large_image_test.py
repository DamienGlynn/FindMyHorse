import cv2
import time
import skimage.io as io
from dglynn.Helper             import Helper
from dglynn.Hog                import Hog
from dglynn.HorseFeatureVector import HorseFeatureVector
from dglynn.HorseClassifier    import HorseClassifier


data_path = 'datasets/data_filtered.h5'
imge_path = 'images/test_images_large/1.jpg'

if __name__ == "__main__":
	
	# Load the data from file.
	datasets = HorseFeatureVector.load(data_path)
	
	X, y = datasets[0], datasets[1]     # Split the data into X samples and y labels.
	horseClassifier = HorseClassifier() # Initialize the Horse classifier.
	horseClassifier.train (X, y)        # Train the classifier.
	
	image = io.imread(imge_path, flatten=False) # Load the image
	image = Helper.setMaxSize(image, 800) # Set the max size of the image.
	(winW, winH) = (64, 128)              # Define the window width and height.
	
	found = []                            # To store the positive predictions.
	
	for resized in Helper.pyramid(image, scale=1.5): # Loop over the image pyramid.
		
		resized = Helper.BGRtoRGB(resized) # Convert the cv2 BRG image to RGB.		
		predictions = [] # Stores the predictions for the current image scale.
		
		# Loop over the sliding window for each layer of the pyramid.
		for (x, y, window) in Helper.sliding_window(resized, stepSize=16, windowSize=(winW, winH)):
			
			# If the window does not meet our desired window size, ignore it.
			if window.shape[0] != winH or window.shape[1] != winW:
				continue
			
			window = window[:,:,-1] # Convert to a 2d array.
			horse_descriptor = Hog.describe(window) # Calculate the hog descriptor for the current window.
			horse_descriptor = horse_descriptor.reshape(1, -1)  # Convert to a 2d array.
			predicted = horseClassifier.predict(horse_descriptor) # Make the predition.
			if predicted[0] > 0: # If there is a match.
				c = (0, 255, 0)
				# Add the predicted window to the predictions list
				predictions.append([x, y, x+winW, y+winH, resized.shape[:2]])
			else:
				c = (0, 0, 255)

			# Display the process
			clone = resized.copy()
			for rect in predictions:
				cv2.rectangle(clone, (rect[0], rect[1]), (rect[0] + winW, rect[1] + winH), (0, 255, 0), 1)
			cv2.rectangle(clone, (x, y), (x + winW, y + winH), c, 1)
			cv2.imshow("Window", clone)
			cv2.waitKey(1)
			time.sleep(0.025)
		

		pick = Helper.non_max_suppression(predictions, 0.2) # Get the bet pick from the overlapping boxes.
		found.extend(pick)                                  # Add the best picks to the found list.

		print( 'Predictions:', len(predictions) )
		print( 'Best Pick:', len(pick) )

		# Display the best predictions at the current image scale.
		# for (x, y, w, h, size) in pick:
		# 	cv2.rectangle(resized, (x, y), (w, h), (0, 255, 0), 1)
		# cv2.imshow("After NMS", resized)
		# cv2.waitKey(0)

	# Show all the best predictions on the original image.
	for rect in found:
		cv2.rectangle(image, (rect[0], rect[1]), ( (rect[0] + rect[2] ), ( rect[1] + rect[3]) ), (0, 255, 0), 1)
	image = Helper.BGRtoRGB(image) # Convert the cv2 BGR image to RGB
	cv2.imshow("Window", image)
	cv2.waitKey(0)

	print( 'Found:', len(found) )

	# # Get the bet pick from the overlapping boxes.
	# pick = Helper.non_max_suppression(found, 0.3)
	# for rect in pick:
	# 	cv2.rectangle(image, (rect[0], rect[1]), ( (rect[0] + rect[2] ), ( rect[1] + rect[3]) ), (0, 255, 0), 1)
	# cv2.waitKey(0)

