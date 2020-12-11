# Ty Bergstrom
# remove_duplicates.py
# September 2020
# CSCE A401
# Software Engineering Project

# Input a dataset and find duplicates.
# Generate hashes for each image to find duplcate hashes.
# Important for ML projects because duplicates can cause bias, and you can get a lot of duplicates from scraping etc.
#
# python3 remove_duplicates.py -d ../original_dataset/mask -r tru
# python3 remove_duplicates.py -d ../original_dataset/without_mask -r tru
#
# optional arg -s to display the duplicates for assurance
# optional arg -r to actually remove duplicates for safety
#
# This is only set up to process one directory at a time, which is a good thing.
# Use preprocess.sh to easily run it on all the directories you need
#
# Note: I made it so that while a duplicate is being displayed,
# you have like 3 seconds to press the "s" key to pass deleting it


from imutils import paths
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True)
ap.add_argument("-r", "--remove", type=bool, default=False)
ap.add_argument("-s", "--show", type=bool, default=False)
args = vars(ap.parse_args())

img_paths = list(paths.list_images(args["dataset"]))

if len(img_paths) < 1:
	print("Err: The directory", args["dataset"] + str(len(img_paths)) , "was empty")
	sys.exit(1)

hashes = {} # dictionary of hashes of all images
hash_size = 8
total_duplicates = 0

# First part, loop through the input images and get their hashes
print("Generating hashes...")
for img_path in img_paths:
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, (hash_size + 1, hash_size))
	# Compute a horizontal gradient between adjacent column pixels
	diff = img[:, 1:] > img[:, :-1]
	# Convert the difference image to a hash
	img_hash = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
	# Find any other image paths with the same hash and add the current image
	paths = hashes.get(img_hash, [])
	# And store the list of paths back in the hashes dictionary
	paths.append(img_path)
	hashes[img_hash] = paths

# Second part, loop through the hashes and find duplicates
print("Finding duplicates...")
for (img_hash, hashed_paths) in hashes.items():
	# Is there more than one image at this place in the dictionary, these have the same hash
	if len(hashed_paths) > 1:
		# Display the duplicates
		if args["show"]:
			montage = None
			for path in hashed_paths:
				image = cv2.imread(path)
				image = cv2.resize(image, (150, 150))
				if montage is None:
					montage = image
				else:
					montage = np.hstack([montage, image])
			cv2.imshow("Duplicates", montage)
			# You have waitKay(5) much time to press "s" key to pass deleting
			if cv2.waitKey(5) == ord("s"):
				print("Duplicate image", path, "was not deleted")
				continue
		# Remove the duplicates
		if args["remove"]:
			for path in hashed_paths[1:]:
				os.remove(path)
				total_duplicates += 1
				print("Deleted duplicate image:", path)
            # End remove all duplicates of this one hash
        # End if removing
    # End if there are duplicates of this one hash
# End loop thru all hashes

print(total_duplicates, "duplicates were removed")



##
