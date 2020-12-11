# Ty Bergstrom
# tuning.py
# CSCE A401
# August 2020
# Software Engineering Project
#
# Hypertuning for any model
# Makes more efficient testing different parameters for increased accuracy
# Choose these parameters from terminal args or for loops
# Instead of crazy editing and commenting out all over the place


from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adadelta
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.optimizers import SGD
from nets.net import Mobile_Net
from nets.net import Quick_Net
from nets.net import Full_Net
from collections import OrderedDict


class Tune:

    # Choose among the neural net implementations and built it
    def build_model(mod, HXW, channels, kernel, num_classes):
        models = [
            "Quick_Net",
            "Full_Net",
			"Mobile_Net"
        ]
        if mod == models[1]:
            return Full_Net.build(width=HXW, height=HXW, depth=channels, kernel=kernel, classes=num_classes)
        if mod == models[2]:
            return Mobile_Net.build(width=HXW, height=HXW, depth=channels, kernel=kernel, classes=num_classes)
        return Quick_Net.build(width=HXW, height=HXW, depth=channels, kernel=kernel, classes=num_classes)


    # Pre-set optimizers for learning rates
    def optimizer(opt, epochs):
        optimizers = OrderedDict([
	        ("SGD",  SGD(lr=0.001, decay=0.1, momentum=0.01, nesterov=True)),
	        ("SGD2", SGD(lr=0.001, decay=0.1, momentum=0.05, nesterov=True)),
	        ("SGD3", SGD(lr=0.001, decay=0.1, momentum=0.1, nesterov=True)),
	        ("Adam",  Adam(lr=0.001, decay=0.001/epochs)),
	        ("Adam2", Adam(lr=0.001, decay=0.001/epochs, amsgrad=True)),
	        ("Adam3", Adam(lr=0.001, beta_1=0.9, beta_2=0.999)),
	        ("Adam4", Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)),
	        ("RMSprop",  RMSprop(lr=0.001, rho=0.9)),
	        ("Adadelta", Adadelta(lr=1.0, rho=0.9)),
        ])
        if opt in optimizers:
            return optimizers[opt]
        return optimizers["Adam"]


    # Pre-set batch sizes (*too large will crash)
    def batch_size(size):
        batch_sizes = [
            16, 24, 32,
            42, 48,
            64, 72, 128
        ]
        if size in batch_sizes:
            return size
        return 32


    # Pre-set image sizes to feed to the model (*too large will crash)
    def img_size(size):
        img_sizes = [
            24, 32,
            48,
            64, 72
        ]
        if size in img_sizes:
            return size
        return 32


    # Pre-set kernel sizes, don't want anything else than 3 or 5
    def kernel(size):
        kernels = [
            3, 5
        ]
        if size in kernels:
            return size
        return 3


    # A valid range of epochs
    def epoch(epochs):
        if epochs >= 1 and epochs <= 200:
            return epochs
        return 1


    # Call to Keras fit_generator() library
    def fit(model, aug, num_epochs, bs, train_X, train_Y, test_X, test_Y):
        hist = model.fit_generator(
            aug.flow(train_X, train_Y, batch_size=bs),
            validation_data=(test_X, test_Y),
            steps_per_epoch=len(train_X) // bs,
            epochs=num_epochs,
            class_weight=[1.0,1.0],
            shuffle=False,
            verbose=2
        )
        return hist


    # Customized learning rates, not currently supported
    #from keras.optimizers import schedules
    def lr_sched(decay, epochs):
        def step_decay(epoch):
            init_lr = 0.1
            drop = 0.5
            epoch_drop = 10.0
            lr = init_lr * math.pow(drop, math.floor((1 + epoch) / epoch_drop))
            return lr
        def poly_decay(epoch):
	        maxEpochs = epochs
	        init_lr = 0.1
	        power = 1.1
	        alpha = init_lr * (1 - (epoch / float(maxEpochs))) ** power
	        return alpha
        if decay == "step":
            return [LearningRateScheduler(step_decay)]
        elif decay == "polynomial":
            return [LearningRateScheduler(poly_decay)]
        elif decay == "reduce":
            lr = ReduceLROnPlateau(
                monitor='val_acc', patience=5,
                verbose=1, factor=0.5, min_lr=0.0001
            )
            return [lr]



    # Pruning the model - currently not supported
    try:
        # Compatibility issues
        import tensorflow_model_optimization as tfmot
    except:
        pass
    def prune(model):
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=2000,
            end_step=4000
        )
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
            model,
            block_size=(1,1),
            block_pooling_type ='AVG',
            pruning_schedule=pruning_schedule
        )
        return pruned_model



##
