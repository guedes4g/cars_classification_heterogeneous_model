from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from readFiles import read_files
import numpy as np

from random import shuffle , seed

import datetime
import pickle

seed(0)
np.random.seed(0)

base_dir = "/home/ml/datasets/DeepLearningFilesPreProcessed/"
test_ratio = 20


def create_classifier():
	clsf = svm.NuSVC(gamma='scale')
	clsf.set_params(nu=0.4, degree=5)
	return clsf

def build_encoder():
	OH = OneHotEncoder(sparse=False)
	return OH 

def split_train_test(classes, OH):
	_X = []
	_Test = []
	for family in classes:
		current = classes[family]
		shuffle(current)
		length = int(len(current))
		train_size = int((length * (100 - test_ratio)) / 100)
		_X += (current[0:train_size])
		_Test += (current[train_size : length])
	shuffle(_X)
	shuffle(_Test)
	X = [np.squeeze(o["features"], axis=0) for o in _X]
	X_Test = [np.squeeze(o["features"], axis=0) for o in _Test]
	y = ([o["label"] for o in _X])
	y_Test = ([o["label"] for o in _Test])
	return np.array(X), np.array(X_Test), np.array(y), np.array(y_Test)

def print_expected_predicted(exp, pred):
	for i in range(len(exp)):
		print("%s :%s - %s"%(exp[i] == pred[i], exp[i], pred[i]))

def main():
	lista, classes = read_files(base_dir)
	OH = build_encoder()
	_max = 0.817917888563049
	X, X_Test, y, y_Test = split_train_test(classes, OH)
	classifier = create_classifier()
	classifier.fit(X, y)
	preds = classifier.predict(X_Test)
	# print_expected_predicted(y, preds)
	current = accuracy_score(y_Test, preds)
	print(current)
	if(_max < current ):
		print( 'accuracy_score %s' %current)
		with open('./models/model%s.pkl'%(datetime.datetime.now()), 'wb') as f:
			pickle.dump(classifier, f)
	# print(_max)

if __name__ == '__main__':
	main()