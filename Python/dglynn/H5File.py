import h5py
import numpy as np

class H5File (object):
	
	def save(path, data):
		''' Save to file. '''
		with h5py.File(path, 'w') as hf:
			for i in range(len(data)):
				hf.create_dataset('vector_%d' % i, data=data[i])
	
	def load(path):
		''' Load from file. '''
		datasets = []
		with h5py.File(path, 'r') as hf:
			for key in hf.keys():
				datasets.append( np.array(hf.get(key)) )
		return datasets