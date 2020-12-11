#!/bin/bash

# Ty Bergstrom
# false_positives.sh
# September 2020
# CSCE A401
# Software Engineering Project
#
# $ bash false_positives.sh
#
# The results from process_faces.py will give you a few images of false positives,
# Where it extracted something else besides the face.
# So you have to manually look through the processed_dataset and find those.
# And you could just delete them or you could go back to the original_dataset,
# Find the originals and crop the face yourself.
#
# Lets find those originals with the following script.
# Based on first manually moving the false positives to the directory below


cd false_positives

dataset="without_mask"
dir_path="../data/masks_hiQ/original_dataset/""${dataset}"
return_path="../../../../processing/double_take/"

# Get all the false positives into an array
falses_arr=( * )

cd ..

# Go to the original_dataset
cd "${dir_path}"

# Loop through the false positives
for val in "${!falses_arr[@]}"
do
	# Eh, need rebuild the original file names 
	# For example from image_967_1.jpg to image_967.jpg
	filename=$(basename -- "${falses_arr[$val]}")
	ext="${filename##*.}"
	original="${filename%\_*}"."${ext}"
	echo "${original}"
	original_img=$(ls | grep ""${original}"")
	echo "${dataset}"_"${original_img}"
	# Copy the originals over to the dir of images that need manual processing
	# Commented out for safety
	#cp "${original_img}" "${return_path}"/"${dataset}"_"${original_img}"
done

# Clean out this dir
# Commented out for safety, prolly not even gonna need to run this again
#rm -rf ../../../../processing/false_positives/*



##
