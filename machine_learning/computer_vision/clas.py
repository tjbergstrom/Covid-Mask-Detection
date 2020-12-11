# Ty Bergstrom
# clas.py
# CSCE A401
# October 2020
# Software Engineering Project
#
# Use my models to make classifications and detections
# Ready to use for both images and videos


from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pickle
import cv2


class Mask_Detector:
	def load_model(path):
		model_path = path + ".model"
		model = load_model(model_path)
		lb_path = path + ".pickle"
		lb = pickle.loads(open(lb_path, "rb").read())
		HXW_reqs = path + "_hxw_req.txt"
		f = open(HXW_reqs, 'r')
		HXW = f.readline()
		HXW = int(HXW)
		f.close()
		return (model, lb, HXW)


	def load_face_detector(proto, model):
		face_detector = cv2.dnn.readNetFromCaffe(proto, model)
		return face_detector


	# Show an image of a face to the mask model
	def predict(frame, box, model):
		(model, lb, HXW) = model
		(start_x, start_y, end_x, end_y) = box
		face = frame[start_y:end_y, start_x:end_x]
		if face.shape[0] < 16 or face.shape[1] < 16:
			return None, None
		face = cv2.resize(face, (HXW, HXW))
		face = face.astype("float") / 255.0
		face = img_to_array(face)
		face = np.expand_dims(face, axis=0)
		prob = model.predict(face)[0]
		idx = np.argmax(prob)
		prediction = lb.classes_[idx]
		probability = prob[idx]
		return prediction, probability


	def mask(lb):
		return lb.classes_[0]


	def without(lb):
		return lb.classes_[1]


	def m_color():
		return (25, 240, 0)


	def wm_color():
		return (32, 0, 255)


##
