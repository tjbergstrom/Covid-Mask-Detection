# Ty Bergstrom
# results.py
# CSCE A401
# August 2020
# Software Engineering Project
#
# Display metrics and plots for any model
# The classification report and confusion matrix will be output to terminal
# And saved to a file performance.txt along with all of the build parameters
# (Every build is appended to the same file)
# Also save a plot


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras import metrics
import numpy as np
import pickle
import time
import matplotlib
matplotlib.use("Agg")


class Result:

	# Save the model with classes after training
	def save_model(model, path, lb, HXW):
		model_path = path + ".model"
		model.save(model_path)
		lb_path = path + ".pickle"
		f = open(lb_path, "wb")
		f.write(pickle.dumps(lb))
		f.close()
		HXW_reqs = path + "_hxw_req.txt"
		f = open(HXW_reqs, "w+")
		f.write(str(HXW))
		f.close()


    # Print the classification report and confusion matrix
	def display_metrix(test_X, test_Y, predictions, model, classes, aug):
		cl = Result.clas_report(test_Y, predictions, classes)
		cm = Result.confusion(model, aug, test_X, test_Y, predictions)
		print("...classification report\n")
		print(cl)
		print("...confusion matrix\n")
		print(cm)
		print()
		Result.save_results(cl, cm)


	def clas_report(test_Y, predictions, classes):
		cl = classification_report(
			test_Y.argmax(axis=1),
			predictions.argmax(axis=1),
			target_names=classes
		)
		return cl


	def confusion(model, aug, test_X, test_Y, predictions):
		pred_idxs = model.predict_generator(aug.flow(test_X, test_Y))
		pred_idxs = np.argmax(pred_idxs, axis=1)
		return confusion_matrix(test_Y.argmax(axis=1), predictions.argmax(axis=1))


	def acc_score(test_Y, predictions):
		return accuracy_score(test_Y.argmax(axis=1), predictions.argmax(axis=1))


	def display_plot(plot, epochs, hist):
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(np.arange(0, epochs), hist.history["loss"], label="train_loss")
		plt.plot(np.arange(0, epochs), hist.history["val_loss"], label="val_loss")
		plt.plot(np.arange(0, epochs), hist.history["acc"], label="train_acc")
		plt.plot(np.arange(0, epochs), hist.history["val_acc"], label="val_acc")
		plt.title("Training Loss & Accuracy")
		plt.xlabel("Epoch number")
		plt.ylabel("Loss/Acc")
		plt.legend(loc="center right")
		plt.savefig(plot)


	# Save the classification report and confusion matrix to file
	def save_results(cl, cm):
		f = open("performance.txt","a+")
		f.write(cl)
		f.write("\n")
		f.write(np.array2string(cm))
		f.write("\n\n\n")
		f.close()


	# Save the build info and parameters to the file
	def save_info(start_time, acc, model, epochs, opt, aug, imgsz, bs, k, datasize, plot, notes):
		run_time = time.time() - start_time

		label_builder = [
			"build: {} ", "{:.6}%\n",
			"model: {}, ", "epochs: {}, ", "optimzer: {}, ", "augmentation: {}\n",
			"image size: {}, ", "batch size: {}, ", "kernel size: {}, ", "dataset size: {}\n",
			"run time: {:.2f}s, ", "{}\n",
			"notes: {} \n"
		]
		labels_str = ""
		for lbl in label_builder:
			labels_str += lbl

		label = labels_str.format(
			start_time, acc*100,
			model, epochs, opt, aug,
			imgsz, bs, k, datasize,
			run_time, plot,
			notes
		)

		f = open("performance.txt","a+")
		f.write(label)
		f.write("\n")
		f.close()



##
