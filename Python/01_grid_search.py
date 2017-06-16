''' This script helps select the best classifier that separates the data. '''
import time

from dglynn.HorseFeatureVector    import HorseFeatureVector
from dglynn.HorseClassifierSearch import HorseClassifierSearch

def time_block(func):
    def wrapper(*args, **kwargs):
        beg_ts = time.time()
        retval = func(*args, **kwargs)
        end_ts = time.time()
        print(time.strftime("%H:%M:%S", time.gmtime(end_ts - beg_ts)))
        return retval
    return wrapper

@time_block
def gridSearchAndTimeIt(X, y):
	# Search for the best classifier.
	return HorseClassifierSearch.gridSearch(X, y)

if __name__ == "__main__":

	data_path = 'datasets/data_filtered.h5'

	# Load the data from file.
	dataset = HorseFeatureVector.load(data_path)

	# Split the data into X samples and y labels.
	X, y = dataset[0], dataset[1]

	# Search for the best classifier and time it.
	precision, recall = gridSearchAndTimeIt(X, y)

	# Plot the decision surface for the classifiers.
	HorseClassifierSearch.plot(precision, X, y)
	HorseClassifierSearch.plot(recall,    X, y)