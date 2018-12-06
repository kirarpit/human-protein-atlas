#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:42:42 2018

@author: Arpit
"""
from keras import initializers
from keras.layers import Dense, Dropout, Flatten, Conv2D, LeakyReLU
from keras.layers import MaxPooling2D, Input, ELU, BatchNormalization
from keras.models import Model, load_model
from keras import optimizers
from keras.utils import multi_gpu_model
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import backend as K
import tensorflow as tf

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

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    
    return focal_loss_fixed

def get_inception_model(params):
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
    
    return model
    
def get_conv_layer(x, filters, kernel_size, bn=True, pool=True, drop=True):
    x = Conv2D(filters = filters, kernel_size = kernel_size,
               padding = 'same', use_bias=False, activation='linear',
               kernel_initializer=initializers.glorot_uniform())(x)
    x = ELU()(x)
    
    if bn:
        x = BatchNormalization(axis=-1)(x)
        
    if pool:
        x = MaxPooling2D(pool_size=(4,4))(x)
        
    if drop:
        x = Dropout(0.25)(x)
    
    return x

def get_fc_layer(x, size):
    x = Dense(size, use_bias=False, activation='linear',
              kernel_initializer=initializers.glorot_uniform())(x)
    x = ELU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.50)(x)
    
    return x
    
def get_simple_model(params):
    input_layer = Input(shape = (*params['dim'], params['n_channels']))
    x = input_layer
    x = get_conv_layer(x, 32, (3,3))
    x = get_conv_layer(x, 64, (3,3), pool=False, drop=False)
    x = get_conv_layer(x, 64, (3,3))
    x = get_conv_layer(x, 128, (3,3), pool=False, drop=False)
    x = get_conv_layer(x, 128, (3,3))
    
    x = Flatten()(x)
    x = get_fc_layer(x, 1024)
    x = get_fc_layer(x, 256)

    output_layer = Dense(params['n_classes'], activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

def get_model(params):
    if os.path.exists('model.h5'):
        print("Loading existing model")
        return load_model('model.h5',  custom_objects={'f1': f1,
                                                       'focal_loss': focal_loss})

    model = get_simple_model(params)
    print(model.summary())
        
    parallel_model = multi_gpu_model(model)
    
    adam = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, 
                           epsilon=None, decay=0.0, amsgrad=False)
    parallel_model.compile(optimizer=adam,
                           loss=[focal_loss(alpha=.25, gamma=2)],
                           metrics=["categorical_accuracy", f1])
    
    return parallel_model