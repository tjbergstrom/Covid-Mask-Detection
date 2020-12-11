# Ty Bergstrom
# trackable_object.py
# CSCE A401
# October 2020
# Software Engineering Project
#
# A trackable_object stores information about each face
# Such as the classification label, the location,
# And what color to draw the bounding box around it.
# update_cnt = reclassify the face after every n frames.
# ready_classify = make first prediction after first n frames.


class Trackable_Object:
	def __init__(self, objectID, centroid):
		self.objectID = objectID
		self.centroids = [centroid]
		self.counted = False
		self.label = ""
		self.color = (0, 0, 0)
		self.update_cnt = 0
		self.ready_classify = 16



##
