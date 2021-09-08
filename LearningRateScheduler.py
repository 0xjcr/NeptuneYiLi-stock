'''
How to choose learning rate with Neptune
'''
###### Create Neptune project 
## update: pip install neptune-client==0.10.7
import neptune
import os

# Connect your script to Neptune
project = neptune.init(api_token=os.getenv('NEPTUNE_API_TOKEN'),
                       project_qualified_name='YourUserName/YourProjectName') 


### from sklearn.datasets import load_iris 
import pandas as pd
import numpy as np
import tensorflow.keras
from tensorflow.keras import backend as K
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras
from tensorflow.keras.layers import Dense, Input, Flatten, concatenate, MaxPooling2D, Conv2D
from tensorflow.keras.models import Model,Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback, LearningRateScheduler 

from sklearn.metrics import f1_score
from tensorflow import random_normal_initializer
from numpy import argmax
from collections import Counter

import os
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
pd.options.display.max_columns = 100

#### Random seed
def reset_random_seeds(CUR_SEED=1234):
   os.environ['PYTHONHASHSEED']=str(CUR_SEED)
   tf.random.set_seed(CUR_SEED)
   np.random.seed(CUR_SEED)
   random.seed(CUR_SEED)

reset_random_seeds()

#### Load data for the image classifier model
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test_full, y_test_full) = fashion_mnist.load_data()

reset_random_seeds()
trainIdx = random.sample(range(60000), 20000)

x_train, y_train = X_train_full[trainIdx]/255.0, y_train_full[trainIdx]
x_test, y_test = X_test_full/255.0, y_test_full
    
#### Save learning rate during the training    
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        curLR = optimizer._decayed_lr(tf.float32)
        return curLR # use ._decayed_lr method instead of .lr
    return lr

### Function to plot the learning rate 
def plotLR(history):
    learning_rate = history.history['lr']
    epochs = range(1, len(learning_rate) + 1)
    fig = plt.figure(figsize=(12, 7))
    plt.plot(epochs, learning_rate)
    plt.title('Learning rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning rate')
    return(fig)

#### Define the Neural Network model
def runModel():   
    model = Sequential()
    model.add(Flatten(input_shape=[28, 28])) 
    model.add(Dense(512, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

model = runModel()
model.summary()


### Learning Rate Schedulers ###
import math
### in the console printout, we can see the learning rate difference  
initial_learning_rate = 0.01
epochs = 100
decay = initial_learning_rate / epochs


CURRENT_LR_SCHEDULER = 'constant'
# CURRENT_LR_SCHEDULER = 'time-based'
# CURRENT_LR_SCHEDULER = 'step-based'
# CURRENT_LR_SCHEDULER = 'exponential'

# CURRENT_LR_SCHEDULER, POLY_POWER = 'polynomial', 'linear'
# CURRENT_LR_SCHEDULER, POLY_POWER = 'polynomial', 'square-root'

# CURRENT_LR_SCHEDULER = 'adaptive'

### Functions to plot the train history 
def plotPerformance(history, CURRENT_LR_SCHEDULER=CURRENT_LR_SCHEDULER):
    #### Loss
    fig = plt.figure(figsize=(10, 4))
    fig = plt.subplot(1, 2, 1) # row 1, col 2 index 1

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Train Loss', 'Test Loss'])
    plt.title(f'Loss Curves ({CURRENT_LR_SCHEDULER})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss on the Validation Set')
    
    #### Accuracy 
    fig = plt.subplot(1, 2, 2) # row 1, col 2 index 1

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['Train Accuracy', 'Test Accuracy'])
    plt.title(f'Accuracy Curves ({CURRENT_LR_SCHEDULER})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy on the Validation Set')
    return fig
    
    
    
    
if CURRENT_LR_SCHEDULER == 'constant':
    # Create an experiment and log the model 
    npt_exp = project.create_experiment(name='ConstantLR', 
                                        description='constant-lr', 
                                        tags=['LearingRate', 'constant', 'baseline', 'neptune'])       
        
    ### Baseline model: constant learning rate 
    initial_learning_rate = 0.01
    epochs = 100
    sgd = keras.optimizers.SGD(learning_rate=initial_learning_rate)
    lr_metric = get_lr_metric(sgd)
    
    model.compile(optimizer = sgd,
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy', lr_metric]) 
    
    reset_random_seeds()
    
    trainHistory_constantLR = model.fit(
        x_train, y_train, 
        epochs=epochs,
        validation_data=(x_test, y_test),
        batch_size=64
    )
    
    ### Plot learning rate over time 
    npt_exp.log_image('Learning Rate Change (Constant)', plotLR(trainHistory_constantLR))
    
    ### Plot the training history 
    npt_exp.log_image('Training Performance Curves (Constant)', plotPerformance(trainHistory_constantLR).get_figure())
    
elif CURRENT_LR_SCHEDULER == 'Keras-buildin':
    # Create an experiment and log the model 
    npt_exp = project.create_experiment(name='KerasBuildInDecay', 
                                        description='Keras-standard-lr-decay', 
                                        tags=['LearningRate', 'standard', 'decay', 'neptune'])
    
    initial_learning_rate = 0.1
    epochs = 100
    
    sgd = keras.optimizers.SGD(learning_rate=initial_learning_rate, decay=0.01) ## decay=0.001, momentum=0.9;
    lr_metric = get_lr_metric(sgd)
    
    model.compile(
                  optimizer = sgd,
                  loss='sparse_categorical_crossentropy', 
                  metrics=[lr_metric]) 
    
    reset_random_seeds()
    
    trainHistory_buildInDecay = model.fit(
        x_train, y_train, 
        epochs=epochs,
        validation_split=0.2, 
        ### training set = 20000*0.8 = 1600 and batch size = 64 -> 250 iterations to complete one epoch, which also means that 
        ### the learning rate will be updated 250 times for one epoch
        batch_size=64
    )

elif CURRENT_LR_SCHEDULER == 'time-based':
    ## initial learning rate set to a larger number 
    initial_learning_rate = 0.5 
    epochs = 100
    decay = initial_learning_rate/epochs   

    # Create an experiment and log the model 
    npt_exp = project.create_experiment(name='TimeBasedLRDecay', 
                                        description='time-based-lr-decay', 
                                        tags=['LearningRate', 'timebased', 'decay', 'neptune'])       

    def lr_time_based_decay(epoch, lr):
        return lr * 1 / (1 + decay * epoch)
    
    model = runModel()
    model.summary()
    
    sgd = keras.optimizers.SGD(learning_rate=initial_learning_rate) 
    model.compile(
                  optimizer = sgd,
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy']) 
    
    reset_random_seeds()
    
    trainHistory_timeBasedDecay = model.fit(
        x_train, y_train, 
        epochs=epochs, 
        batch_size=64,
        validation_split=0.2,
        callbacks=[LearningRateScheduler(lr_time_based_decay, verbose=1)])    
    
    ### Plot learning rate over time 
    npt_exp.log_image('Learning Rate Change (Time-Based Decay)', plotLR(trainHistory_timeBasedDecay))
    ### Plot the training history 
    npt_exp.log_image('Training Performance Curves (Time-Based Decay)', plotPerformance(trainHistory_timeBasedDecay).get_figure())
    
elif CURRENT_LR_SCHEDULER == 'step-based': ### aka Discrete Staircase Decay 
    initial_learning_rate = 0.5  
    epochs = 100
    decay = initial_learning_rate/epochs   
    
    # Create an experiment and log the model 
    npt_exp = project.create_experiment(name='StepBasedLRDecay', 
                                        description='step-based-lr-decay', 
                                        tags=['LearningRate', 'stepbased', 'decay', 'neptune'])       

    def lr_step_based_decay(epoch):
        drop_rate = 0.8 
        epochs_drop = 10.0
        return initial_learning_rate * math.pow(drop_rate, math.floor(epoch/epochs_drop))
    
    model = runModel()
    model.summary()
    
    sgd = keras.optimizers.SGD(learning_rate=initial_learning_rate) 
    model.compile(
                  optimizer = sgd,
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy']) 
    
    reset_random_seeds()
    
    trainHistory_stepBasedDecay = model.fit(
        x_train, y_train, 
        epochs=epochs, 
        batch_size=64,
        validation_split=0.2,
        callbacks=[LearningRateScheduler(lr_step_based_decay, verbose=1)])    
    
    ### Plot learning rate over time 
    npt_exp.log_image('Learning Rate Change (Step-Based Decay)', plotLR(trainHistory_stepBasedDecay))
    ### Plot the training history 
    npt_exp.log_image('Training Performance Curves (Step-Based Decay)', plotPerformance(trainHistory_stepBasedDecay).get_figure())

    
elif CURRENT_LR_SCHEDULER == 'exponential':
    initial_learning_rate = 0.5 
    epochs = 100
    decay = initial_learning_rate/epochs   
    
    # Create an experiment and log the model 
    npt_exp = project.create_experiment(name='ExponentialLRDecay', 
                                        description='exponential-lr-decay', 
                                        tags=['LearningRate', 'exponential', 'decay', 'neptune'])
    
    def lr_exp_decay(epoch):
        k = 0.1
        return initial_learning_rate * math.exp(-k*epoch)

    model = runModel()
    model.summary()
    
    sgd = keras.optimizers.SGD(learning_rate=initial_learning_rate) 
    model.compile(
                  optimizer = sgd,
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy']) 
    
    reset_random_seeds()
    
    trainHistory_expDecay = model.fit(
        x_train, y_train, 
        epochs=epochs, 
        batch_size=64,
        validation_split=0.2,
        callbacks=[LearningRateScheduler(lr_exp_decay, verbose=1)])    
    
    ### Plot learning rate over time 
    npt_exp.log_image('Learning Rate Change (Exponential Decay)', plotLR(trainHistory_expDecay))
    ### Plot the training history 
    npt_exp.log_image('Training Performance Curves (Exponential Decay)', plotPerformance(trainHistory_expDecay).get_figure())

elif CURRENT_LR_SCHEDULER == 'polynomial':
    initial_learning_rate = 0.5 
    epochs = 100
    decay = initial_learning_rate/epochs   
    
    ## Defined as a class to save parameters as attributes
    class lr_polynomial_decay:
    	def __init__(self, epochs=100, initial_learning_rate=0.01, power=1.0):
    		# store the maximum number of epochs, base learning rate, and power of the polynomial
    		self.epochs = epochs
    		self.initial_learning_rate = initial_learning_rate
    		self.power = power
            
    	def __call__(self, epoch):
    		# compute the new learning rate based on polynomial decay
    		decay = (1 - (epoch / float(self.epochs))) ** self.power
    		updated_eta = self.initial_learning_rate * decay
    		# return the new learning rate
    		return float(updated_eta)
        
    def plot_Neptune(history, decayTitle):
        ### Plot learning rate over time 
        npt_exp.log_image(f'Learning Rate Change ({decayTitle})', plotLR(history))
        ### Plot the training history 
        npt_exp.log_image(f'Training Performance Curves ({decayTitle})', plotPerformance(history).get_figure())
    
    POLY_POWER = 'linear'
    # POLY_POWER = 'square-root'
    # POLY_POWER = 'square'
    
    # Create an experiment and log the model 
    npt_exp = project.create_experiment(name=f'{POLY_POWER}LRDecay', 
                                        description=f'{POLY_POWER}-lr-decay', 
                                        tags=['LearningRate', POLY_POWER, 'decay', 'neptune'])   
       
    if POLY_POWER == 'linear':
        curPower = 1.0
    elif POLY_POWER == 'square-root':
        curPower = 0.5
    elif POLY_POWER == 'square':
        curPower = 2.0
   
    curScheduler = lr_polynomial_decay(epochs=epochs, initial_learning_rate=initial_learning_rate, power=curPower)
    
    model = runModel()
    model.summary()
    
    sgd = keras.optimizers.SGD(learning_rate=initial_learning_rate) 
    model.compile(
                  optimizer = sgd,
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy']) 
    
    reset_random_seeds()
    
    trainHistory_polyDecay = model.fit(
        x_train, y_train, 
        epochs=epochs, 
        batch_size=64,
        validation_split=0.2,
        callbacks=[LearningRateScheduler(curScheduler, verbose=1)]) 
    
    if POLY_POWER == 'linear':
        trainHistory_linearDecay = trainHistory_polyDecay
        plot_Neptune(history=trainHistory_linearDecay, decayTitle='Linear Decay')
    elif POLY_POWER == 'square-root':
        trainHistory_sqRootDecay = trainHistory_polyDecay
        plot_Neptune(history=trainHistory_sqRootDecay, decayTitle='SquareRoot Decay')
    elif POLY_POWER == 'square':
        trainHistory_squaredDecay = trainHistory_polyDecay
        plot_Neptune(history=trainHistory_squaredDecay, decayTitle='Squared Decay')
    
elif CURRENT_LR_SCHEDULER == 'adaptive':
    # Create an experiment and log the model 
    npt_exp = project.create_experiment(name='Adaptive', 
                                        description='adaptive-lr', 
                                        tags=['LearningRate', 'adam', 'neptune'])    
    
    model = runModel()
    model.summary()
    
    ## adam = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    adam = keras.optimizers.Adam()
    lr_metric = get_lr_metric(adam)
    model.compile(
                  optimizer=adam,
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy', lr_metric]) 
    
    reset_random_seeds()
    
    trainHistory_adaptive = model.fit(
        x_train, y_train, 
        epochs=100, 
        batch_size=64,
        validation_split=0.2)
    
    plot_Neptune(history=trainHistory_adaptive, decayTitle='Adam Optimizer')


    
#=========== Create an experiment and log the model
npt_exp = project.create_experiment(name='ModelComparison', 
                                    description='compare-lr-schedulers', 
                                    tags=['LearningRate', 'schedulers', 'comparison', 'neptune'])  

def masterComparePlot(metric, ylab, plotTitle, NeptuneImageTitle, includeAdaptive=False, npt_exp=npt_exp, subset=True):
    if subset:
        num = 60
    else:
        num = len(trainHistory_timeBasedDecay.history['val_loss'])
    
    fig = plt.figure(figsize=(12, 7))
    plt.plot(trainHistory_constantLR.history[metric][:num])
    plt.plot(trainHistory_timeBasedDecay.history[metric][:num])
    plt.plot(trainHistory_stepBasedDecay.history[metric][:num])
    plt.plot(trainHistory_expDecay.history[metric][:num])
    plt.plot(trainHistory_linearDecay.history[metric][:num])
        
    if not includeAdaptive:          
        plt.legend(['Constant LR', 'Time-based Decay', 'Step-based Decay', 'Exponential Decay', 
                    'linear Decay'])        
    else:
        plt.plot(trainHistory_adaptive.history[metric][:num])
        plt.legend(['Constant LR', 'Time-based Decay', 'Step-based Decay', 'Exponential Decay', 
                    'linear Decay', 'Adam'])
        
    plt.ylabel(ylab)
    plt.xlabel('Epochs')
    plt.title(plotTitle)
        
    npt_exp.log_image(NeptuneImageTitle, fig)

    
###### Compare loss decay curves 
masterComparePlot('val_loss', ylab='Loss on the Validation Set', plotTitle='Compare Validation Loss', 
                  NeptuneImageTitle='Compare Model Performance -- Loss', includeAdaptive=False)

###### Compare the Accuracy curves
masterComparePlot('val_accuracy', ylab='Accuracy on the Validation Set', plotTitle='Compare Validation Accuracy', 
                  NeptuneImageTitle='Compare Model Performance -- Accuracy', includeAdaptive=False)

###### Compare LR decay curves 
masterComparePlot('lr', ylab='Learning Rate', plotTitle='Compare Learning Rate Curves Generated from Different Schedulers', 
                  NeptuneImageTitle='Compare Learning Rate Curves', includeAdaptive=False, subset=False)


if CURRENT_LR_SCHEDULER == 'adaptive':    
    ##### Compare SGD with Decay vs. ADAM 
    ###### Compare loss decay curves 
    masterComparePlot('val_loss', ylab='Loss on the Validation Set', plotTitle='Compare Validation Loss', 
                      NeptuneImageTitle='Compare Model Performance -- Loss', includeAdaptive=True)
    
    ###### Compare the Accuracy curves
    masterComparePlot('val_accuracy', ylab='Accuracy on the Validation Set', plotTitle='Compare Validation Accuracy', 
                      NeptuneImageTitle='Compare Model Performance -- Accuracy', includeAdaptive=True)
    
    ###### Compare LR decay curves 
    masterComparePlot('lr', ylab='Learning Rate', plotTitle='Compare Learning Rate Curves Generated from Different Schedulers', 
                      NeptuneImageTitle='Compare Learning Rate Curves', includeAdaptive=True, subset=False)
    
























