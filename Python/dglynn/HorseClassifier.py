from sklearn import svm

class HorseClassifier (object):
	''' This class trains a Support Vector Classifier on a feature vector X, y.
		It can then predict which class a sample is in. 
		And return the decision value of the prediction '''
	def __init__(self):
		super(HorseClassifier, self).__init__()
		# The best classifier calculated from the GridSearchCV.
		# self.clf = svm.SVC(kernel='rbf', C=100, gamma=0.001)
		#self.clf = svm.SVC(kernel='linear', C=1)
		self.clf = svm.SVC(kernel='poly', C=100, degree=5, coef0=1)

	def train(self, X, y):
		''' Train the Support Vector Classifier on the X, y feature vector. '''
		self.clf.fit(X, y)

	def predict(self, sample):
		''' Make a prediction on the given sample. '''
		return self.clf.predict(sample)

	def decision(self, sample):
		''' Get the decision value '''
		return self.clf.decision_function(sample)

