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
    
class SVHN(Dataset.Dataset):

    def __init__(self, smaller_data_size=None, normalize=True, data_path=None):
        self.name = 'svhn'

        self.n_classes = 10
        self.data_path = data_path
        if data_path is not None:
            self.train_file = os.path.join(data_path, "train_32x32.mat")
            self.test_file = os.path.join(data_path, "test_32x32.mat")
        else:
            self.train_file = "datasets/train_32x32.mat"
            self.test_file = "datasets/test_32x32.mat"
            
        super(SVHN, self).__init__(normalize=normalize)
    
    def load_training_data(self):
        mat = spio.loadmat(self.train_file, squeeze_me=True)
        x_train = mat["X"]
        y_train = mat["y"] - 1
        x_train = np.moveaxis(x_train, -1, 0)
        del mat
        y_train_labels = one_hot_encoded(y_train, num_classes=self.n_classes)
        return x_train, np.array(y_train), np.array(y_train_labels)

    def load_test_data(self):
        mat = spio.loadmat(self.test_file, squeeze_me=True)
        x_test = mat["X"]
        y_test = mat["y"] - 1
        del mat
        x_test = np.moveaxis(x_test, -1, 0)
        y_test_labels = one_hot_encoded(y_test, num_classes=self.n_classes)
        return x_test, np.array(y_test), np.array(y_test_labels)

    def normalize_dataset(self):
        pass

    def maybe_download(self):
        pass
