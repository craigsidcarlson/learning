from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from scipy.spatial import distance
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import numpy as np
import csv
import os
import sys

features = []
labels = []

test_idx = [40, 50, 349]

file = sys.argv[1]


with open(os.path.splitext(file)[0] +'_FEATURES.csv', 'rb') as featureFile:
		featurereader = csv.DictReader(featureFile)
		try:
			for feature in featurereader:
				features.append(feature)
		except csv.Error as e:
			sys.exit('file %s, line %d: %s' % (filename, featurereader.line_num, e))

with open(os.path.splitext(file)[0] +'_LABELS.csv', 'rb') as labelFile:
		labelreader = csv.reader(labelFile)
		labelreader.next()#Skip the header
		for label in labelreader:
			labels.append(label)


le = preprocessing.LabelEncoder()
np.ravel(labels)
# print labels.shape
y = le.fit_transform(labels)

feature_vec = DictVectorizer()
X = feature_vec.fit_transform(features).toarray()
print y
print y.shape
print X.shape
train_feature, test_feature, train_label, test_label = train_test_split(X, y, test_size = .25)


classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(train_feature, train_label)


predicted = classifier.predict(test_feature)

result = le.inverse_transform(predicted)
print 'tested %s records' % len(test_label)
print 'accuracy %s' % accuracy_score(le.inverse_transform(test_label), result)
