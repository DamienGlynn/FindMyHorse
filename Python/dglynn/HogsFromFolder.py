import os
import numpy as np
import skimage.io as io
from dglynn.Hog import Hog

class HogsFromFolder (object):
	
	def extract(path):
		''' Returns a numpy array of hog descriptors from a folder of images. '''
		hogs  = []
		for file in os.listdir(path):
			file = io.imread(path + '/' + file, flatten=True)
			hog_descriptor = Hog.describe(file)
			hogs.append(hog_descriptor)
		return np.array(hogs)