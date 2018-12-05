#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:42:42 2018

@author: Arpit
"""
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras import optimizers
from keras.utils import multi_gpu_model
from keras.applications.inception_resnet_v2 import InceptionResNetV2

def get_model(params):
    net = InceptionResNetV2(include_top=False,
                          weights='imagenet',
                          input_tensor=None,
                          input_shape=(512, 512, 3))
    x = net.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    
    output_layer = Dense(params['n_classes'], activation='sigmoid', name='sigmoid')(x)
    model = Model(inputs=net.input, outputs=output_layer)
    
    for layer in model.layers[:-2]:
        layer.trainable = False
    for layer in model.layers[-2:]:
        layer.trainable = True
        
    parallel_model = multi_gpu_model(model, gpus=2)
    
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, 
                           epsilon=None, decay=0.0, amsgrad=False)
    parallel_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    return parallel_model