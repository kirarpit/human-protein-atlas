#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 20:47:29 2018

@author: Arpit
"""

from keras import optimizers
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from data_generator import DataGenerator

from utils import get_data_ids, get_labels, split_ids

# Parameters
params = {'dim': (512,512),
          'batch_size': 64,
          'n_classes': 28,
          'n_channels': 3,
          'shuffle': True}

# Datasets
training_ids, testing_ids = get_data_ids()
training_ids, validation_ids = split_ids(training_ids, 0.85)
labels = get_labels()

# Generators
training_generator = DataGenerator(training_ids, labels, **params)
validation_generator = DataGenerator(validation_ids, labels, **params)

# Design model
net = InceptionResNetV2(include_top=False,
                          weights='imagenet',
                          input_tensor=None,
                          input_shape=(3, 512, 512))
x = net.output
x = Flatten()(x)
x = Dropout(0.5)(x)

output_layer = Dense(params['n_classes'], activation='sigmoid', name='sigmoid')(x)
model = Model(inputs=net.input, outputs=output_layer)

for layer in model.layers[:-5]:
    layer.trainable = False
for layer in model.layers[-5:]:
    layer.trainable = True


adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, 
                       epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

print(model.summary())

# Train
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)

# Predict
#preds = model.predict(X_test)
#preds[preds>=0.5] = 1
#preds[preds<0.5] = 0