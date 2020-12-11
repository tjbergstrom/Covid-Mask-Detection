# Ty Bergstrom
# process_faces.py
# September 2020
# CSCE A401
# Software Engineering Project
#
# Input original images.
# Detect and extract a face or faces from each image.
# Save each face as a new 256x256 image (originals are not overwritten)
# Find them in the directory processed_dataset/
#
# This is for if the models need good photos of just faces and
# need to exclude everything else from an image.
# Make sure to check the processed dataset afterwords and
# delete any images that aren't actually a face,
# or later you can manually crop the faces on any that aren't perfect
#
# Other notes:
# Any images that did not detect a face are saved in processing/double_take/
# So that you can find them and manually process those images much quicker.
#
# This is set up to only process one directory at a time, there's a good reason.
# Use preprocess.sh to run this on each directory that you need
#
# "--originals_dir" is the parent dir of the original dataset
# "--dataset" is the sub dir of the classes
# "--processed_dir" is the new parent dir where the extracted faces will be saved
#
# python3 process_faces.py -d mask
# python3 process_faces.py -d without_mask -o tru
# But it's better run run from a script for consistency

# The -n arg is to limit each image to extract only one face


from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import sys
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True)
ap.add_argument("-o", "--originals_dir", default="../data/original_dataset/")
ap.add_argument("-p", "--processed_dir", default="../data/processed_dataset/")
ap.add_argument("-n", "--one", type=bool, default=False)
args = vars(ap.parse_args())

proto = "../prebuilt_detectors/deploy.prototxt"
model = "../prebuilt_detectors/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(proto, model)
threshold = 0.8

processed_dir = args["processed_dir"]
originals_dir = args["originals_dir"]
dataset = args["dataset"]

img_paths = sorted(list(paths.list_images(originals_dir + dataset)))

if len(img_paths) < 1:
	print("Err: The directory", originals_dir + dataset, "was empty")
	sys.exit(1)

total_saved = 0
one_photo_per_img = args["one"]
HXW = 256

print("processing:", len(img_paths), "input images from", originals_dir + dataset)

for (itr, img_path) in enumerate(img_paths):
	image = cv2.imread(img_path)
	(h, w) = image.shape[:2]

	# Detect faces in the image
	img_blob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)),
		1.0, (300, 300),
		(104.0, 177.0, 123.0),
		swapRB=False, crop=False
	)
	detector.setInput(img_blob)
	detections = detector.forward()

	# Append this iterator to the save filepath if more than one face per image
	img_itr = 0

    # Loop through all detected faces in the image
	for i in range(0, detections.shape[2]):
		if one_photo_per_img:
			i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]
		if confidence > threshold:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(start_x, start_y, end_x, end_y) = box.astype("int")
			face = image[start_y:end_y, start_x:end_x]
			too_small = 32
			if face.shape[0] < too_small or face.shape[1] < too_small:
				continue
			face = imutils.resize(face, width=HXW, height=HXW)
			filename = os.path.basename(img_path)
			filename, ext = os.path.splitext(filename)
			filename = processed_dir + dataset + "/" + filename + "_" + str(img_itr) + ext
			# If you were renaming here with this script, you would construct this new filename
			#filename = processed_dir + dataset + "/" + str(itr) + "_" + str(img_itr) + ".jpg"
			cv2.imwrite(filename, face)
			total_saved += 1
			img_itr += 1
			if one_photo_per_img:
				break
		# End if probability for this face is above threshold
	# End for every face detected in the image

	if img_itr == 0:
		# Did not extract a face from this image, so save a copy
		# So you can quickly find it and manually process it later
		filename = "double_take/" + dataset + "_" + os.path.basename(img_path)
		cv2.imwrite(filename, image)

# End looping thru every image in the dataset

print("saved", total_saved, "extracted faces images")



##
