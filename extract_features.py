import os
import cv2
import keras.backend as K
from utils import load_model
import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image, ExifTags

cv2.IMREAD_IGNORE_ORIENTATION = 0


def resize_and_pad(im):
	desired_size = 224
	old_size = im.shape[:2] # old_size is in (height, width) format
	ratio = float(desired_size)/max(old_size)
	new_size = tuple([int(x*ratio) for x in old_size])
	# new_size should be in (width, height) format
	im = cv2.resize(im, (new_size[1], new_size[0]))
	delta_w = desired_size - new_size[1]
	delta_h = desired_size - new_size[0]
	top, bottom = delta_h//2, delta_h-(delta_h//2)
	left, right = delta_w//2, delta_w-(delta_w//2)
	color = [125, 125, 125]
	new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
	    value=color)
	return new_im

def rotate_image(path):
	from PIL import Image, ExifTags
	image=Image.open(path)
	try:
	    for orientation in ExifTags.TAGS.keys():
	        if ExifTags.TAGS[orientation]=='Orientation':
	            break
	    exif=dict(image._getexif().items())

	    if exif[orientation] == 3:
	        image=image.rotate(180, expand=True)
	    elif exif[orientation] == 6:
	        image=image.rotate(270, expand=True)
	    elif exif[orientation] == 8:
	        image=image.rotate(90, expand=True)
	    # image.save(path)
	    # image.close()

	except (AttributeError, KeyError, IndexError):
	    # cases: image don't have getexif
	    pass
	return np.array(image)

def load_convnet():
	model = load_model()
	model.summary()
	return model

def find_paths(path):
	paths = []
	labels = []
	level_a = os.listdir(path)
	for level_name in level_a:
		for image_name in os.listdir(os.path.join(path, level_name)):
			labels += [level_name]
			paths += [os.path.join(path, level_name, image_name)]

	return paths, labels

def transform_image(im_data):
	# im_data = cv2.imread(path)	
	return resize_and_pad(im_data)

def augment(img, label):
	aug_seq = iaa.Sequential(
        [
            iaa.Add((-20, 20)),
            iaa.ContrastNormalization((0.8, 1.6)),
            iaa.AddToHueAndSaturation((-21, 21)),
            iaa.SaltAndPepper(p=0.1),
        ],
	random_order=True)

	images = []
	labels = []
	for i in range(10):
		images += [aug_seq.augment_image(img)]
		labels += [label]

	return images, labels


def create_folder(path):
	print("creating folder: %s"%path)
	try:
		os.mkdir(path)
	except Exception as e:
		print('folder already exists')
		pass

def main():
	net = load_convnet()
	get_features = K.function([net.layers[0].input, K.learning_phase()], [net.get_layer("flatten_2").output])
	base_path = "/home/ml/datasets/DeepLearningFiles"
	paths, labels = find_paths(base_path)
	base_path_save = "/home/ml/datasets/DeepLearningFilesPreProcessed/"
	create_folder(base_path_save)
	
	for i, (image_path , label) in enumerate(zip(paths, labels)):
		try:
			print("%s %s %s"%("base", label, i))
			# returns rgb
			im_data = rotate_image(os.path.join(base_path, image_path))
			im_data = transform_image(im_data)
			aug_images, aug_labels = augment(im_data, label)
			im_data = im_data/255.
			features = get_features([im_data[np.newaxis,...],0])[0]
			np.savez_compressed("%s%s_%s.npz"%(base_path_save, label, i),**{"f1":features, "label": label, "base_id": i, "base": True})
			for j, (aug_image, aug_label) in enumerate(zip(aug_images, aug_labels)):
				aug_image = aug_image/255.
				print("%s %s %s %s "%("aug", label, i, j))
				features = get_features([aug_image[np.newaxis,...],0])[0]
				np.savez_compressed("%s%s_%s_%s_aug.npz"%(base_path_save,aug_label, i, j),**{"f1":features, "label": aug_label, "base_id": i, "base": False})
		except Exception as e:
			print(e)
			print(label, image_path)


if __name__ == "__main__":
	main()