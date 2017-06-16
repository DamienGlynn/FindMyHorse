from skimage.feature import hog

class Hog (object):
	
	def describe(image, orientations=9, pixels_per_cell=8, cells_per_block=2, visualise=False):
		''' Returns the feature descriptor and feature image for a given image '''
		return hog(image,
				   orientations    = orientations,
				   pixels_per_cell = (pixels_per_cell, pixels_per_cell),
				   cells_per_block = (cells_per_block, cells_per_block),
				   visualise       = visualise)





