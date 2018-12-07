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
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    
    return focal_loss_fixed
    
def get_conv_layer(x, filters, kernel_size, bn=True, pool=True, drop=True):
    x = Conv2D(filters = filters, kernel_size = kernel_size,
               padding = 'same', activation='linear',
               kernel_initializer=initializers.glorot_uniform())(x)
    x = ELU()(x)
    
    if bn:
        x = BatchNormalization(axis=-1)(x)
        
    if pool:
        x = MaxPooling2D(pool_size=(4,4))(x)
        
    if drop:
        x = Dropout(0.25)(x)
    
    return x

def get_fc_layer(x, size, bn=True):
    x = Dense(size, activation='linear',
              kernel_initializer=initializers.glorot_uniform())(x)
    x = ELU()(x)

    if bn:
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

def get_inception_model(params):
    input_tensor = Input(shape=(*params['dim'], 3))
    x = BatchNormalization()(input_tensor)

    net = InceptionResNetV2(include_top=False,
                          weights='imagenet',
                          input_tensor=None,
                          input_shape=(*params['dim'], 3))

    x = net(x)
    x = get_conv_layer(x, 256, (1,1), bn=False, pool=False, drop=False)
    x = get_conv_layer(x, 64, (1,1), bn=False, pool=False, drop=False)
    x = Flatten()(x)
    x = Dropout(0.5)(x)

    x = get_fc_layer(x, 1024, bn=False)
    x = get_fc_layer(x, 256, bn=False)

    output_layer = Dense(params['n_classes'], activation='sigmoid')(x)
    model = Model(inputs=input_tensor, outputs=output_layer)

    return model

def get_model(params, inception=False):
    if os.path.exists('model.h5'):
        print("Loading existing model")
        return load_model('model.h5',  custom_objects={'f1': f1, 'f1_loss': f1_loss})

    if inception:
        model = get_inception_model(params)
        model.layers[2].trainable = True
    else:
        model = get_simple_model(params)
        
    print(model.summary())
        
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(optimizer=optimizers.Adam(lr=1e-4),
                           loss=[f1_loss],
#                           loss=['binary_crossentropy'],
                           metrics=["categorical_accuracy", f1])
    
    return parallel_model
