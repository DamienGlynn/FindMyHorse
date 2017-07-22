import numpy      as np
import skimage.io as io

from dglynn.HogsFromFolder     import HogsFromFolder
from dglynn.HorseFeatureVector import HorseFeatureVector

'''
1: Extract the Hog descriptors from the positive and negetive folders.
2: Create the HorseFeatureVector dataset from the Hog descriptors.
3: Save the HorseFeatureVector dataset to a H5 file.
'''

print('Extracting hog descriptors...')
pos_X = HogsFromFolder.extract('images/horses/new/all_high')
neg_X = HogsFromFolder.extract('images/negative_images_x2')

print(pos_X.shape, neg_X.shape)

print('Creating feature vectors...')
X, y = HorseFeatureVector.create(pos_X, neg_X)

print('Saving to file...')
HorseFeatureVector.save('datasets/data_new.h5', X, y)
print('Done!')