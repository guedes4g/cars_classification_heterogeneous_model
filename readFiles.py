import numpy as np 
import os


def list_files(path):
	return os.listdir(path)

def read_files(base_path):
	file_names = list_files(base_path)
	files = []
	classes = {}
	for i , file_name in enumerate(file_names):

		path = os.path.join(base_path, file_name)
		with np.load(path) as file:
			data = {
			"file_name": file_name,
			"features" : file['f1'],
			"base" :  file['base'].tolist(),
			"base_id": file['base_id'].tolist(),
			"label" : file['label'].tolist()
			}
			files += [data]

			if data['label'] in classes:
				classes[data['label']] += [data]
			else:
				classes[data['label']] = [data]

	print("Done reading...")
	return files , classes

