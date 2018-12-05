#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 20:47:29 2018

@author: Arpit
"""

from data_generator import DataGenerator
from model import get_model
from utils import get_data_ids, get_labels, split_ids

# Parameters
params = {'dim': (512,512),
          'batch_size': 64,
          'n_classes': 28,
          'n_channels': 3,
          'shuffle': True,
          'dir_path': 'data/training_data/'}

# Datasets
training_ids, testing_ids = get_data_ids()
training_ids, validation_ids = split_ids(training_ids, 0.85)
labels = get_labels()

# Generators
training_generator = DataGenerator(training_ids, labels, **params)
validation_generator = DataGenerator(validation_ids, labels, **params)

# Design model
model = get_model(params)
print(model.summary())

# Train
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6,
                    epochs=1,
                    verbose=1)

model.save('model')

# Predict
params['shuffle'] = False
params['batch_size'] = 1
params['dir_path'] = 'data/testing_data/'
testing_generator = DataGenerator(testing_ids, labels=None, **params)

preds = model.predict_generator(testing_generator,
                                steps=len(testing_ids),
                                use_multiprocessing=False,
                                workers=1,
                                verbose=1)

print(preds, preds.shape, len(testing_ids))