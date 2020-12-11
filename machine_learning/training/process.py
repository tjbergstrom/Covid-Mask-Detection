# Ty Bergstrom
# process.py
# CSCE A401
# August 2020
# Software Engineering Project
#
# Pre-process the dataset for any model


from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import random
import cv2
import sys
import os


class Pprocess:

	# Load the input dataset and process
	def preprocess(dataset, HXW):
		data = []
		cl_labels = []
		img_paths = sorted(list(paths.list_images(dataset)))
		if len(img_paths) < 1:
			print("Err: No images found in:", dataset)
			sys.exit(1)
		random.seed(64)
		random.shuffle(img_paths)
		for img_path in img_paths:
			img = cv2.imread(img_path)
			img = cv2.resize(img, (HXW, HXW))
			img = img_to_array(img)
			data.append(img)
			label = img_path.split(os.path.sep)[-2]
			cl_labels.append(label)
		data = np.array(data, dtype="float") / 255.0
		if len(data) != len(cl_labels):
			print("Err: Data not correctly labeled with classes")
			sys.exit(1)
		return data, cl_labels


	# Processing required for the class labels to work
	def class_labels(cl_labels):
		lb = LabelBinarizer()
		cl_labels = lb.fit_transform(cl_labels)
		num_classes = len(lb.classes_)
		if num_classes < 2:
			print("Err: Not enough classes")
			sys.exit(1)
		return lb, cl_labels, num_classes


	# Loss type depending on how many classes
	def binary_or_categorical(num_classes):
		if num_classes > 2:
			return "categorical_crossentropy"
		return "binary_crossentropy"


	# sklearn train test split, with extra processing for binary classification
	def split(data, cl_labels, num_classes, test_size=0.2):
		cl_labels = np.array(cl_labels)
		(train_X, test_X, train_Y, test_Y) = train_test_split(
			data, cl_labels,
			test_size=test_size,
			random_state=64
		)
		if num_classes == 2:
			train_Y = to_categorical(train_Y, num_classes=num_classes)
			test_Y = to_categorical(test_Y, num_classes=num_classes)
		return train_X, test_X, train_Y, test_Y


	# Different options for augmentation pre-processing
	# Too much is bad, so these are rolled back to small incremental changes
	def data_aug(aug):
		wsr = 0.1
		hsr = 0.1
		if aug == "light1":
			return ImageDataGenerator(horizontal_flip=True, fill_mode="nearest",
            rotation_range=15, shear_range=0.1, zoom_range=0.1,
            width_shift_range=wsr, height_shift_range=hsr)
		if aug == "light2":
			return ImageDataGenerator(horizontal_flip=True, fill_mode="reflect",
            rotation_range=15, shear_range=0.1, zoom_range=0.1,
            width_shift_range=wsr, height_shift_range=hsr)
		if aug == "light3":
			return ImageDataGenerator(horizontal_flip=True, fill_mode="wrap",
            rotation_range=15, shear_range=0.1, zoom_range=0.1,
            width_shift_range=wsr, height_shift_range=hsr)

		if aug == "medium1":
			return ImageDataGenerator(horizontal_flip=True, fill_mode="nearest",
            rotation_range=30, shear_range=0.2, zoom_range=0.2,
            width_shift_range=wsr, height_shift_range=hsr)
		if aug == "medium2":
			return ImageDataGenerator(horizontal_flip=True, fill_mode="reflect",
            rotation_range=30, shear_range=0.2, zoom_range=0.2,
            width_shift_range=wsr, height_shift_range=hsr)
		if aug == "medium3":
			return ImageDataGenerator(horizontal_flip=True, fill_mode="wrap",
            rotation_range=30, shear_range=0.2, zoom_range=0.2,
            width_shift_range=wsr, height_shift_range=hsr)

		if aug == "heavy1":
			return ImageDataGenerator(horizontal_flip=True, fill_mode="reflect",
            rotation_range=45, shear_range=0.2, zoom_range=0.3,
            width_shift_range=wsr, height_shift_range=hsr)
		if aug == "heavy2":
			return ImageDataGenerator(horizontal_flip=True, fill_mode="nearest",
            rotation_range=45, shear_range=0.2, zoom_range=0.3,
            width_shift_range=wsr, height_shift_range=hsr)
		if aug == "heavy3":
			return ImageDataGenerator(horizontal_flip=True, fill_mode="wrap",
            rotation_range=45, shear_range=0.2, zoom_range=0.3,
            width_shift_range=wsr, height_shift_range=hsr)

		else:
			return ImageDataGenerator(horizontal_flip=True, fill_mode="nearest",
            rotation_range=30, shear_range=0.2, zoom_range=0.2,
            width_shift_range=wsr, height_shift_range=hsr)



##
