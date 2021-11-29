#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from .. import download
import sys
from six.moves import cPickle
from keras import backend as K
from ..datasets.Dataset import one_hot_encoded
from ..datasets import Dataset

import numpy as np
import scipy.io as spio
    
class EMNIST(Dataset.Dataset):
    def __init__(self, smaller_data_size=None, normalize=True, data_path=None):
        self.name = 'emnist'

        self.n_classes = 10
        self.data_path = data_path
        if data_path is not None:
            self.train_file = os.path.join(data_path, "emnist-digits.mat")
            self.test_file = os.path.join(data_path, "emnist-digits.mat")
        else:
            self.train_file = "datasets/emnist-digits.mat"
            self.test_file = "datasets/emnist-digits.mat"
        print("load emnist data from {} and {}".format(self.train_file, self.test_file))
            
        super(EMNIST, self).__init__(normalize=normalize)
    
    def load_training_data(self):
        mat = spio.loadmat(self.train_file)
        x_train = mat["dataset"][0][0][0][0][0][0]
        x_train = x_train.astype(np.float32)
        x_train /= 255
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1, order="A")
        y_train = np.squeeze(mat["dataset"][0][0][0][0][0][1])
        print("x_train shape: {}, y_train shape: {}".format(x_train.shape, y_train.shape))
        
        del mat
        y_train_labels = one_hot_encoded(y_train, num_classes=self.n_classes) # create one-hot encoding
        return x_train, np.array(y_train), np.array(y_train_labels)

    def load_test_data(self):
        mat = spio.loadmat(self.test_file)
        x_test = mat["dataset"][0][0][1][0][0][0]
        x_test = x_test.astype(np.float32)
        x_test /= 255
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1, order="A")
        y_test = np.squeeze(mat["dataset"][0][0][1][0][0][1])
        print("x_test shape: {}, y_test shape: {}".format(x_test.shape, y_test.shape))

        del mat
        y_test_labels = one_hot_encoded(y_test, num_classes=self.n_classes)
        return x_test, np.array(y_test), np.array(y_test_labels)

    def normalize_dataset(self):
        pass

    def maybe_download(self):
        pass
