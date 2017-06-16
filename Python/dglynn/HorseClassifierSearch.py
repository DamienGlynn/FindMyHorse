import numpy  as np
from matplotlib              import pyplot as plt
from sklearn.decomposition   import PCA
from sklearn                 import svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV

class HorseClassifierSearch (object):
	''' This class searches for the best classifier that fits the data. '''

	def gridSearch(X, y):
		''' Performs an extensive search of the data using various parameters
			for a Support Vector Classifier (SVC) and returns two SVCs with the best
			parameters that fit the data for 1:precision and 2:recall. '''
		train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.5, random_state=0)
		tuned_parameters = [{'kernel': ['rbf'],    'C': [1, 10, 100, 1000], 'gamma': [1e-3, 1e-4]},
		                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
		                    {'kernel': ['poly'],   'C': [1, 10, 100, 1000], 'degree': [2, 5], 'coef0': [0, 1]}]
		scores = ['precision', 'recall']
		for score in scores:
		    print("\nTuning hyper-parameters for %s\n" % score)
		    clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5, scoring='%s_macro' % score)		    
		    clf.fit(train_X, train_y)
		    print("\nBest parameters set found on development set")
		    print(clf.best_params_, '\n')
		    means = clf.cv_results_['mean_test_score']
		    stds = clf.cv_results_['std_test_score']
		    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		        print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, params))
		    print("\nDetailed classification report:")
		    y_true, y_pred = test_y, clf.predict(test_X)
		    print(metrics.classification_report(y_true, y_pred))
		    if score == 'precision':
		    	precision_clf = svm.SVC(**clf.best_params_)
		    else:
		    	recall_clf    = svm.SVC(**clf.best_params_)
		return precision_clf, recall_clf

	def plot(clf, X, y):
		''' Plot the data and display the classifier hyper-plane. '''
		X_train, _, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=0)
		pca = PCA(n_components=2).fit(X_train)
		pca_2d = pca.transform(X_train)
		svmClassifier_2d = clf.fit(pca_2d, y_train)
		for i in range(0, pca_2d.shape[0]):
			if y_train[i] == 0:
				c1 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='r', s=50, marker='+')
			elif y_train[i] == 1:
				c2 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='g', s=50, marker='o')
		plt.legend([c1, c2], ['Negative', 'Positve'])
		x_min, x_max = pca_2d[:, 0].min() - 1, pca_2d[:, 0].max() + 1
		y_min, y_max = pca_2d[:, 1].min() - 1, pca_2d[:, 1].max() + 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, .01), np.arange(y_min, y_max, .01))
		Z = svmClassifier_2d.predict(np.c_[xx.ravel(),  yy.ravel()])
		Z = Z.reshape(xx.shape)
		plt.contour(xx, yy, Z)
		plt.title('Support Vector Machine Decision Surface')
		plt.axis('off')
		mng = plt.get_current_fig_manager()
		mng.window.showMaximized()
		plt.show()




