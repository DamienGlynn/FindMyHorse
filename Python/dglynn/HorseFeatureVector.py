import numpy as np

from matplotlib                import pyplot as plt
from sklearn.decomposition     import PCA
from sklearn.model_selection   import train_test_split
from dglynn.H5File             import H5File

class HorseFeatureVector (object):
	
	def create(pos_vectors, neg_vectors):
		''' Creates a feature vector X, y for given positive and negative vectors '''
		pos_labels = np.ones ((1, len(pos_vectors)))   # Create the positive labels.
		neg_labels = np.zeros((1, len(neg_vectors)))   # Create the negative labels.
		X = np.concatenate((pos_vectors, neg_vectors)) # Conctenate the two vectors.
		y = np.append(pos_labels, neg_labels)          # Append the labels.
		# Randomise the vectors and label keeping the indexes in-sync.
		perm = np.random.permutation(len(X))
		X = X[perm]
		y = y[perm]
		# Return the feature vector.
		return X, y

	def plot(X, y):
		''' Show the feature vector on a 2-d plot. '''
		pca    = PCA(n_components=2).fit(X)
		pca_2d = pca.transform(X)
		for i in range(0, pca_2d.shape[0]):
			if y[i] == 0:
				c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', s=50, marker='+')
			elif y[i] == 1:
				c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', s=50, marker='o')
		plt.legend([c1, c2], ['False', 'True'])
		plt.title('Irish Cob Dataset')
		mng = plt.get_current_fig_manager()
		mng.window.showMaximized()
		plt.axis('off')
		plt.show()	
	
	def save(path, X, y):
		''' Save the feature vector to file. '''
		data = [X, y]
		H5File.save(path, data)

	def load(path):
		''' Load the feature vector from file. '''
		return H5File.load(path)