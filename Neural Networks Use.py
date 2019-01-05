#!python
# NN Demonstration

'''

example cases for several common types of neural networks using keras

demonstrate the creation of several different models in keras
    standard
    CNN

RNN in the form of LSTM cells will be done in a different project
as there are no suitable keras datasets and an outside one will
need to be taken in and processed

'''
import time
import keras
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import TensorBoard
from tensorflow.keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


tensorboard = TensorBoard(log_dir="NNuse_logs/example_network_{}".format(int(time.time())))


# Base

def run_nn():       # direct, feed forward, fully connected neural network

    (x_train, y_train), (x_test, y_test) = mnist.load_data()        # mnist handwritten didgets, for CNN

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    
    model = Sequential()
    model.add(Flatten())    # converts input into a 1d array

    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, verbose=0, callbacks=[tensorboard])

    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(val_loss, val_acc)


# CNN

def run_cnn():      # examins groups of input at a time, effective in image recognition and other areas
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()        # mnist handwritten didgets, for CNN
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)


    model = Sequential()

    model.add(Conv2D(128, (3, 3), activation='relu', input_shape = x_train.shape[1:])) # add in input shapes
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, verbose=0, callbacks=[tensorboard])
    # model fit won't not be verbose

    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(val_loss, val_acc)











