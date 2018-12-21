#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 19:00:02 2018

@author: KushDani
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

mnist = keras.datasets.mnist

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

print(train_data.shape)
print(train_labels[0])
plt.imshow(train_data[0])
plt.colorbar()
plt.grid()

#scale values to be between 0 and 1 and cast to float
#train_data = train_data / 255.0
#test_data = test_data / 255.0

plt.clf()
"""for i in range(10):
    plt.subplot(4,5,i+1) #displays multiple images at once
    #get rid of x and y axis ticks
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(train_labels[i])
    plt.imshow(train_data[i])"""

model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(100, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax),
        
    ])
    

model.compile(optimizer=tf.train.AdamOptimizer(0.0008),
              loss=['sparse_categorical_crossentropy'],
              metrics=['accuracy'])
    
model.fit(train_data, train_labels, epochs=5, verbose=1, validation_split=0.2,
          callbacks=[keras.callbacks.EarlyStopping(monitor='acc',
                                                      patience=3)])

#still working on getting accuracy up without overfitting as I learn more about models and parameter weight
