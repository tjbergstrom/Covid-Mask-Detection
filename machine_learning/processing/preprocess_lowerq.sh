#!/bin/bash

# Ty Bergstrom
# preprocess_lowerq.sh
# September 2020
# CSCE A401
# Software Engineering Project
#
# $ bash preprocess_lowerq.sh
#
# Pre-processing for the lower quality dataset
#
# Running this python file for each class directory
# $ python3 lowerq.py -o ../data/masks_lowQ/original_dataset/mask -p ../data/masks_lowQ/processed_dataset/mask
# And making sure those directories are cleaned out first

# Input path to the directory of cropped hi quality faces images
original_dir="../data/masks_lowQ/original_dataset"
# Outout path to the directory where the low quality processed faces images will go
processed_dir="../data/masks_lowQ/processed_dataset_P"

printf "Checking directories \n"

if [ ! -d "${original_dir}" ]; then
    printf "Err: The directory "${original_dir}" does not exist \n"
    exit 1
fi

if [ ! "$(ls -A "${original_dir}")" ]; then
    printf "Err: The directory "${original_dir}" is empty\n"
    exit 1
fi

printf "Preparing directories \n"

# Make sure the processed directories exist and clean them out

rm -rf "${processed_dir}"
mkdir -p "${processed_dir}"
mkdir -p "${processed_dir}"/mask
mkdir -p "${processed_dir}"/without_mask


printf "Processing... \n"

# Run the python scripts that process the images on each class's dataset

datasets=("mask" "without_mask")

for i in "${!datasets[@]}"
do
    python3 lowerq.py -o "${original_dir}"/"${datasets[$i]}" -p "${processed_dir}"/"${datasets[$i]}"
    printf "Processed: "
    ls "${processed_dir}"/"${datasets[$i]}" | wc -l
done

printf "Pre-processing complete \n\n"



##
