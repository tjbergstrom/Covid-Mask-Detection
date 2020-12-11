# Ty Bergstrom
# train_a_model.py
# CSCE A401
# October 2020
# Software Engineering Project
#
# Train any model with this file
#
# $ source ./venv1/bin/activate
# $ python3 -W ignore train_a_model.py
#
# This pre-processes an input dataset with selected tunings
# Trains a selected model with selected tunungs and saves it
# Outputs results and metrics including a plot and saves them with the build info


from process import Pprocess
from results import Result
from tuning import Tune
import argparse
import time


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--savemodel", type=str, default="model_name")
ap.add_argument("-p", "--plot", type=str, default="plots/plot.png")
ap.add_argument("-d", "--dataset", type=str, default="../data/masks_hiQ/processed_dataset")
ap.add_argument("-m", "--model", type=str, default="Full_Net")
ap.add_argument("-a", "--aug", type=str, default="default")
ap.add_argument("-b", "--batch_size", type=int, default=32)
ap.add_argument("-n", "--notes", type=str, default="(none)")
ap.add_argument("-e", "--num_epochs", type=int, default=50)
ap.add_argument("-k", "--kernel_size", type=int, default=3)
ap.add_argument("-o", "--opt", type=str, default="Adam3")
ap.add_argument("-i", "--img_size", type=int, default=48)
args = vars(ap.parse_args())

batch_size = Tune.batch_size(args["batch_size"])
model_path = "../models/" + args["savemodel"]
num_epochs = Tune.epoch(args["num_epochs"])
kernel = Tune.kernel(args["kernel_size"])
HXW = Tune.img_size(args["img_size"])
notes = args["notes"]
channels = 3
start_time = time.time()

print("\n...pre-processing the data...\n")
(data, cl_labels) = Pprocess.preprocess(args["dataset"], HXW)
(lb, cl_labels, num_classes) = Pprocess.class_labels(cl_labels)
loss_type = Pprocess.binary_or_categorical(num_classes)
(train_X, test_X, train_Y, test_Y) = Pprocess.split(data, cl_labels, num_classes)
aug = Pprocess.data_aug(args["aug"])

print("\n...building the model...\n")
model = Tune.build_model(args["model"], HXW, channels, kernel, num_classes)
opt = Tune.optimizer(args["opt"], num_epochs)
model.compile(loss=loss_type, optimizer=opt, metrics=["accuracy"])

print("\n...training the model...\n")
hist_obj = Tune.fit(model, aug, num_epochs, batch_size, train_X, train_Y, test_X, test_Y)
Result.save_model(model, model_path, lb, HXW)

print("\n...getting results of training & testing...\n")
predictions = model.predict(test_X, batch_size=batch_size)
Result.save_info(start_time, Result.acc_score(test_Y, predictions), args["model"], num_epochs, args["opt"], args["aug"], HXW, batch_size, kernel, len(data), args["plot"], notes)
Result.display_metrix(test_X, test_Y, predictions, model, lb.classes_, aug)
Result.display_plot(args["plot"], num_epochs, hist_obj)



##
