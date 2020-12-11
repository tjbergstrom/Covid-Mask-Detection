# Ty Bergstrom
# detect_and_track.py
# CSCE A401
# October 2020
# Software Engineering Project
#
# Input webcam or file
# Detect faces and make classification
# Track faces and add up statistics
#
# python3 -W ignore detect_and_track.py
# python3 -W ignore detect_and_track.py -d tru
# python3 -W ignore detect_and_track.py -f uploads/samplevid0.mp4


from clas import Mask_Detector
from centroid_tracker import Centroid_Tracker
from trackable_object import Trackable_Object
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import argparse
import time
import os


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--device", default=False)
ap.add_argument("-f", "--filepath", default="uploads/samplevid2.mp4")
ap.add_argument("-m", "--model", default="masks_lowQ_P988713/masks_lowQ")
ap.add_argument("-t", "--testfile", required=False)
ap.add_argument("-o", "--output", required=False)
args = vars(ap.parse_args())

testing = args["testfile"]
if testing:
	test_file = open(args["testfile"], "a")

model = Mask_Detector.load_model(
	"../models/" + args["model"]
)
face_detector = Mask_Detector.load_face_detector(
    "../prebuilt_detectors/deploy.prototxt",
    "../prebuilt_detectors/res10_300x300_ssd_iter_140000.caffemodel"
)

if args["device"]:
	vs = cv2.VideoCapture(0)
	time.sleep(1.0)
else:
	vs = cv2.VideoCapture(args["filepath"])

face_detect_frequency = 4
reclassify_frequency = 32
total_frames = 0
masks_cnt = 0
without_cnt = 0
usage = 0.0

(_, lb, _) = model
mask_label = Mask_Detector.mask(lb)
without_label = Mask_Detector.without(lb)

ct = Centroid_Tracker(max_disappear=32, max_distance=48)
trackers = []
trackable_objects = {}


# Pass a frame thru the face detection net
def detect_faces():
	global trackers
	trackers = []
	blob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)),
		1.0,
		(300, 300),
		(104.0, 177.0, 123.0),
		swapRB=False,
		crop=False
	)
	face_detector.setInput(blob)
	detections = face_detector.forward()
	# Add detected faces to trackers
	for i in range(0, detections.shape[2]):
		probability = detections[0, 0, i, 2]
		if probability < 0.5:
			break
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(start_x, start_y, end_x, end_y) = box.astype("int")
		if (end_x - start_x) < 16 or (end_y - start_y) < 16:
			continue
		tracker = dlib.correlation_tracker()
		box = dlib.rectangle(start_x, start_y, end_x, end_y)
		tracker.start_track(rgb, box)
		trackers.append(tracker)
	# End for detected faces in frame


# Get the new location of a face in the current frame
def track_faces(tracker, boxs, rgb):
	tracker.update(rgb)
	pos = tracker.get_position()
	start_x = int(pos.left())
	start_y = int(pos.top())
	end_x = int(pos.right())
	end_y = int(pos.bottom())
	boxs.append((start_x, start_y, end_x, end_y))
	return boxs


def ready_classify(to):
	return to.ready_classify <= 0


def wearing_mask(prediction):
	return prediction == mask_label


def not_wearing_mask(prediction):
	return prediction == without_label


def detected_mask(prediction, to):
	global masks_cnt
	to.label = prediction
	to.color = Mask_Detector.m_color()
	masks_cnt += 1


def detected_without(prediction, to):
	global without_cnt
	to.label = prediction
	to.color = Mask_Detector.wm_color()
	without_cnt += 1


# Update the counters and the trackable_object based on the prediction
def classify_face(box, to):
	if ready_classify(to):
		prediction, prob = Mask_Detector.predict(frame, box, model)
		if prob is None:
			return
		if prob < 0.25:
			return
	else:
		to.ready_classify -= 1
		return
	if wearing_mask(prediction):
		detected_mask(prediction, to)
		to.counted = True
	elif not_wearing_mask(prediction):
		detected_without(prediction, to)
		to.counted = True


# Check if a mask was removed or put on
def reclassify(box, to):
	global masks_cnt, without_cnt
	try:
		prediction, prob = Mask_Detector.predict(frame, box, model)
		if prob < 0.5:
			return
	except:
		return
	if prediction != to.label:
		if wearing_mask(prediction):
			detected_mask(prediction, to)
			without_cnt -= 1
		elif not_wearing_mask(prediction):
			detected_without(prediction, to)
			masks_cnt -= 1


def draw_stats():
	global usage
	cv2.rectangle(frame, (0,0), (155,55), (0,0,0), -1)
	label = "{}: {}".format(str(mask_label), str(masks_cnt))
	cv2.putText(frame, label, (10, 15), 0, 0.55, Mask_Detector.m_color(), 2)
	label = "{}: {}".format(str(without_label), str(without_cnt))
	cv2.putText(frame, label, (10, 30), 0, 0.55, Mask_Detector.wm_color(), 2)
	if masks_cnt + without_cnt > 0:
		usage = masks_cnt / (masks_cnt + without_cnt)
	label = "usage: {:.2f}%".format(usage * 100)
	cv2.putText(frame, label, (10, 45), 0, 0.55, (222, 0, 222), 2)


def mask_detection(frame):
	global trackers
	boxs = []
	# Scan frame, detect faces, add faces to trackers
	if (total_frames % face_detect_frequency) == 0:
		detect_faces()
	# Track detected faces
	else:
		for tracker in trackers:
			boxs = track_faces(tracker, boxs, rgb)
	# Update new locations of detected faces
	objects = ct.update(boxs)
	# Checkout detected faces
	for (objectID, (centroid, box)) in objects.items():
		to = trackable_objects.get(objectID, None)
		# Add a new face as a trackable object
		if to is None:
			to = Trackable_Object(objectID, centroid)
		# Attempt to classify faces that are being tracked
		else:
			to.centroids.append(centroid)
			# Make first classify ~16 frames after the face was was first detected
			if not to.counted and ct.disappeared[objectID] < 8 and to.label == "":
				classify_face(box, to)
			# Reclassify after every n frames
			elif to.counted and to.update_cnt > reclassify_frequency:
				# Reclassify if the face is being tracked
				if ct.disappeared[objectID] < 16:
					to.update_cnt = 0
					reclassify(box, to)
				# Standby if recently lost track of the face
				elif ct.disappeared[objectID] >= 8 and ct.disappeared[objectID] <= 16:
					to.update_cnt -= 1
				# Do not reclassify if lost track of the face for more than 16 frames
				else:
					to.update_cnt = int(reclassify_frequency / 2)
			# Reject if a face was detected but lost track for the last 8 of its first 16 frames
			elif not to.counted and ct.disappeared[objectID] >= 8:
				ct.disappeared[objectID] = 99
			# End (re)classify conditions
			to.update_cnt += 1
		# End add a new face or classify currently tracking faces
		trackable_objects[objectID] = to
		label = str(to.label)
		(start_x, start_y, end_x, end_y) = box
		txt_y = start_y + int((end_y - start_y) / 6)
		cv2.putText(frame, label, (start_x + 10, txt_y), 0, 0.5, to.color, 2)
		cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), to.color, 1)
	# End for detected faces in frame
	draw_stats()
	return frame



writer = None
while True:
	frame = vs.read()
	frame = frame[1]
	if frame is None:
		break
	frame = imutils.resize(frame, width=500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	(h, w) = frame.shape[:2]
	frame = mask_detection(frame)
	if not testing:
		cv2.imshow("Demo Video", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	if args["output"]:
		if writer is None:
			fps = vs.get(cv2.CAP_PROP_FPS)
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, fps, (w, h), True)
		writer.write(frame)
	total_frames += 1
# End while true reading video


vs.release()
cv2.destroyAllWindows()
print("Mask usage: {:.2f}%".format(usage * 100))

if testing:
	usage = "{:.2f}".format(usage*100)
	f = os.path.basename(args["filepath"])
	test_file.write(
		f + "," + str(int(masks_cnt)) + "," + str(int(without_cnt)) + "," + str(usage) + "\n"
	)



####
