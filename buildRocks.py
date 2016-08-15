from sklearn.cross_validation import train_test_split
import csv
import sys
import os
features = ['TECTONIC SETTING', 'LOCATION', 'ROCK TEXTURE', 'ALTERATION', 'PRIMARY/SECONDARY', 'ROCK NAME', 'GRAIN SIZE']
labels = ['MINERAL']

minerals_features = []
minerals_labels = []



def buildData(file):
	file = sys.argv[1]
	with open(file, 'rb') as readFile:
		reader = csv.DictReader(readFile)

		for row in reader:
			mineral_feature = {}
			for feature in features:
				mineral_feature[feature] = row[feature]
			minerals_features.append(mineral_feature)

			mineral_label = {}
			for label in labels:
				mineral_label[label] = row[label]
			minerals_labels.append(mineral_label)

	with open(os.path.splitext(file)[0] +'_FEATURES.csv', 'w') as featuresFile:
		featurewriter = csv.DictWriter(featuresFile, fieldnames=features)
		featurewriter.writeheader()
		featurewriter.writerows(minerals_features)

	with open(os.path.splitext(file)[0] +'_LABELS.csv', 'w') as labelsFile:
		labelwriter = csv.DictWriter(labelsFile, fieldnames=labels)
		labelwriter.writeheader()
		labelwriter.writerows(minerals_labels)

mineralData = buildData('data/rocks/minerals/CLAY_MINERALS.csv')