from readFiles import read_files
import numpy as np
from sklearn.metrics import accuracy_score
import pickle


feature_files = "/home/ml/datasets/DeepLearningFilesPreProcessed/"
model_path = "/home/ml/code/aula/cars_feature_extractor/models/model2018-12-16 17:15:04.269703.pkl"

def load_model(path):
	with open(path, 'rb') as f:
		return pickle.load(f)

def print_expected_predicted(exp, pred):
	for i in range(len(exp)):
		print("%s :%s - %s"%(exp[i] == pred[i], exp[i], pred[i]))

def main():
	classifier = load_model(model_path)
	_X, classes = read_files(feature_files)
	X = [np.squeeze(o["features"], axis=0) for o in _X]
	y = ([o["label"] for o in _X])
	preds   = classifier.predict(X)
	accuracy = accuracy_score(y, preds) 
	print("accuracy: %s"%(accuracy))


if __name__ == '__main__':
	main()