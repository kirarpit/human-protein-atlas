#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 00:21:55 2018

@author: Arpit
"""
import csv
import numpy as np

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
    np.random.shuffle(l)
    
    training_idxs = l[:int(percent*len(ids))]
    testing_idxs = list(set(l)^set(training_idxs))
    
    training_ids = [ids[index] for index in training_idxs]
    testing_ids = [ids[index] for index in testing_idxs]
    
    return training_ids, testing_ids