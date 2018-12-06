#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 20:47:29 2018

@author: Arpit
"""

from data_generator import DataGenerator
from model import get_model
from utils import get_data_ids, get_labels, split_ids, save_preds, save_numpy
from keras.callbacks import ModelCheckpoint

# Parameters
params = {'dim': (299, 299),
          'batch_size': 16,
          'n_classes': 28,
          'n_channels': 3,
          'shuffle': True,
          'augment': True,
          'dir_path': '/tmp/human_atlas/training_data/'}

# Datasets
training_ids, testing_ids = get_data_ids()
training_ids, validation_ids = split_ids(training_ids, 0.80)
labels = get_labels()

# Generators
training_generator = DataGenerator(training_ids, labels, **params)
params['batch_size'] = 128
validation_generator = DataGenerator(validation_ids, labels, **params)

# Design model
model = get_model(params, True)
print(model.summary())

# Train
checkpointer = ModelCheckpoint(filepath='model.h5', verbose=2,
                               save_best_only=True, save_weights_only=False)
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    callbacks=[checkpointer],
                    epochs=1)

# Predict
params['shuffle'] = False
params['augment'] = False
params['batch_size'] = 1
params['dir_path'] = '/tmp/human_atlas/testing_data/'
testing_generator = DataGenerator(testing_ids, labels=None, **params)

preds = model.predict_generator(testing_generator,
                                steps=len(testing_ids),
                                verbose=1)
save_numpy(preds)
save_preds(preds, testing_ids)
print(preds, preds.shape, len(testing_ids))
