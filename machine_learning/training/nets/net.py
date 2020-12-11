# Ty Bergstrom
# net.py
# CSCE A401
# August 2020
# Software Engineering Project
#
# Neural network implementations


from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.layers.core import Dense
from keras import backend as K


# This is a lightweight net for testing builds and datasets more quickly
class Quick_Net:

	@staticmethod
	def build(width, height, depth, kernel, classes):
		input_shape = (height, width, depth)
		if K.image_data_format() == "channels_first":
			input_shape = (depth, height, width)

		model = Sequential()

		# first set of convolutional layers
		model.add(Conv2D(32, (kernel, kernel), padding="same", input_shape=input_shape))
		model.add(Activation("relu"))
		model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.1))

		# second set of convolutional layers
		model.add(Conv2D(64, (kernel, kernel), padding="same"))
		model.add(Activation("relu"))
		model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.1))

		# only set of fully connected relu layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model



# This is a deeper bigger net for when you are prepared to wait a while
class Full_Net:

	@staticmethod
	def build(width, height, kernel, depth, classes):
		input_shape = (height, width, depth)
		if K.image_data_format() == "channels_first":
			input_shape = (depth, height, width)

		model = Sequential()

		# first set of convolutional layers
		model.add(Conv2D(32, (kernel, kernel), padding="same", input_shape=input_shape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.2))

		# second set of convolutional layers
		model.add(Conv2D(64, (kernel, kernel), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.2))

		# third set of convolutional layers
		model.add(Conv2D(128, (kernel, kernel), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.2))

		# fourth set of convolutional layers
		model.add(Conv2D(264, (kernel, kernel), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.2))

		# only set of fully connected relu layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model



class Mobile_Net:

	try:
		from keras_applications import mobilenet_v2
		#from keras.applications import MobileNetV2
	except:
		pass
	from keras.models import Model

	@staticmethod
	def build(width, height, kernel, depth, classes):
		input_shape = (height, width, depth)
		if K.image_data_format() == "channels_first":
				input_shape = (depth, height, width)

		# Load MobileNet as the base
		base = mobilenet_v2(
			weights="imagenet",
			include_top=False,
			input_tensor=Input(shape=input_shape)
		)

		# Add a custom head to the base with project-specific output classes
		head = base.output
		head = AveragePooling2D(pool_size=(7, 7))(head)
		head = Flatten(name="flatten")(head)
		head = Dense(128, activation="relu")(head)
		head = Dropout(0.5)(head)
		head = Dense(classes, activation="softmax")(head)

		model = Model(inputs=base.input, outputs=head)

		for layer in base.layers:
			layer.trainable = False

		return model



##
