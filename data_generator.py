#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 20:40:05 2018

@author: Arpit
"""

from PIL import Image
import numpy as np
import keras
import cv2

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    
    def __init__(self, list_IDs, labels, batch_size=32, dim, n_channels,
                 n_classes, shuffle=True, dir_path=None):
        'Initialization'
        
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.dir_path = dir_path

        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i] = self.load(ID)

            # Store class
            y[i] = self.get_multiclass_labels(ID)

        return X, y
    
    def load(self, ID):
        
        R = np.array(Image.open(self.dir_path + ID + '_red.png'))/255
        G = np.array(Image.open(self.dir_path + ID + '_green.png'))/255
        B = np.array(Image.open(self.dir_path + ID + '_blue.png'))/255
        Y = np.array(Image.open(self.dir_path + ID + '_yellow.png'))/255
        
        image = np.stack([R/2+Y/2, G, B/2+Y/2], axis=-1)
        
        if self.dim != (512, 512):
            image = cv2.resize(image, self.dim, cv2.INTER_AREA)        
            
        return image
    
    def get_multiclass_labels(self, ID):
        if self.labels is None:
            # in case of predicting
            return np.array([0]*self.n_classes)
        
        multi_label = self.labels[ID]
        multi_lable_encoded = [1 if i in multi_label else 0 for i in range(self.n_classes)]
        
        return np.array(multi_lable_encoded)
