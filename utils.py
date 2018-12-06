#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 00:21:55 2018

@author: Arpit
"""
import csv
import numpy as np
import pickle

def get_data_ids():
    training_ids = []
    with open("train.csv") as file:
        reader = csv.reader(file)
        next(reader, None)

        for row in reader:
            training_ids.append(row[0])
    
    testing_ids = []
    with open("sample_submission.csv") as file:
        reader = csv.reader(file)
        next(reader, None)

        for row in reader:
            testing_ids.append(row[0])
    
    return training_ids, testing_ids

def get_labels():
    d = {}
    
    with open("train.csv") as file:
        reader = csv.reader(file)
        next(reader, None)

        for row in reader:
            d[row[0]] = list(map(int, row[1].split(' ')))

    return d

def split_ids(ids, percent):
    l = np.arange(len(ids))
    np.random.seed(0)
    np.random.shuffle(l)
    
    training_idxs = l[:int(percent*len(ids))]
    testing_idxs = list(set(l)^set(training_idxs))
    
    training_ids = [ids[index] for index in training_idxs]
    testing_ids = [ids[index] for index in testing_idxs]
    
    return training_ids, testing_ids

def save_preds(preds, ids):
    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0
    
    f = open('predictions.csv', 'w')
    f.write("Id,Predicted\n")
    
    for i in range(len(preds)):
        l = list(np.where(preds[i] == 1.0)[0])
        labels = ' '.join(str(e) for e in l)
        f.write(ids[i] + "," + labels + "\n")
    f.close()
    
def save_numpy(filename, data):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def load_numpy(filename):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    
    return data
