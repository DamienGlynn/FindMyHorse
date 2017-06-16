import os
import skimage.io as io

from matplotlib                import pyplot as plt
from dglynn.Helper             import Helper
from dglynn.Hog                import Hog
from dglynn.HorseFeatureVector import HorseFeatureVector
from dglynn.HorseClassifier    import HorseClassifier

data_path = 'datasets/data_filtered.h5'

if __name__ == "__main__":
	
	# Load the data from file.
	datasets = HorseFeatureVector.load(data_path)

	X, y = datasets[0], datasets[1]     # Split the data into X samples and y labels.
	horseClassifier = HorseClassifier() # Initialize the Horse classifier.
	horseClassifier.train (X, y)        # Train the classifier.
	path = 'images/test_images_small/'  # Get the path of the unknown samples.
	
	fig = plt.figure()                  # Create a figure.
	fig.suptitle('Predictions for Unknown Samples', fontsize=20)
	
	position = 1

	# Loop through the folder.
	for file in os.listdir(path):

		# Read in the image.
		image = io.imread(path + '/' + file, flatten=False)
		
		# Create the HOG descriptor.
		hog_descriptor  = Hog.describe(image[:,:, -1])

		# Reshape into a 2-d array.
		hog_descriptor  = hog_descriptor.reshape(1, -1)

		# Make the prediction.
		predicted = horseClassifier.predict(hog_descriptor)

		# Get the actual decision.
		decision = horseClassifier.decision(hog_descriptor)
		
		# Create a subplot.
		ax = fig.add_subplot(2, 6, position)

		# Set the subplots title.
		ax.set_title('Predicted: %s\nDecision: %f' % (predicted[0], decision[0]))

		# Turn the axis off.
		ax.axis('off')

		# Show the image.
		ax.imshow( image, aspect='auto' )

		# Increment the position for the next image.
		position += 1

	# Display the figure in full screen.
	mng = plt.get_current_fig_manager()
	mng.window.showMaximized()

	# Show the figure.
	plt.show()

