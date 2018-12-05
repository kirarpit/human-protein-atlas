#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:42:42 2018

@author: Arpit
"""
from keras.layers import Dense, Dropout, Flatten, Conv2D, LeakyReLU
from keras.models import Model, load_model
from keras import optimizers
from keras.utils import multi_gpu_model
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import backend as K

import os

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_model(params):
    if os.path.exists('model'):
        print("Loading existing model")
        return load_model('model')

    net = InceptionResNetV2(include_top=False,
                          weights='imagenet',
                          input_tensor=None,
                          input_shape=(512, 512, 3))
    x = net.output
    x = Conv2D(filters = 1, kernel_size = (1,1),
               padding = 'same', use_bias=False, activation='linear')(x)
    x = LeakyReLU()(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(196, use_bias=False, activation='linear')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    x = Dense(196, use_bias=False, activation='linear')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    output_layer = Dense(params['n_classes'], activation='sigmoid', name='sigmoid')(x)
    model = Model(inputs=net.input, outputs=output_layer)
    
    """
    Freeze everything till this layer
    conv_7b_ac (Activation)         (None, 14, 14, 1536) 0           conv_7b_bn[0][0]
    """
    trainable = False
    for layer in model.layers:
        layer.trainable = trainable
        if layer.name == 'conv_7b_ac':
            trainable = True
            
    print(model.summary())
        
    parallel_model = multi_gpu_model(model, gpus=2)
    
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, 
                           epsilon=None, decay=0.0, amsgrad=False)
    parallel_model.compile(loss='binary_crossentropy', optimizer=adam, 
                           metrics=["categorical_accuracy", f1])
    
    return parallel_model