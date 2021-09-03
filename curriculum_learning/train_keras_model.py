#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 18:33:53 2018

@author: guy.hacohen
"""
import keras
import numpy as np
import keras.backend as K
import time

def compile_model(model, initial_lr=1e-3, loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'], momentum=0.0):
    if optimizer == 'adam':
        optimizer = keras.optimizers.Adam(initial_lr, beta_1=0.9, beta_2=0.999,
                                          epsilon=None, decay=0.0,
                                          amsgrad=False)
    elif optimizer == 'sgd':
        optimizer = keras.optimizers.SGD(initial_lr, momentum=momentum)
    else:
        print("optimizer not supported")
        raise ValueError
    
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics)


def basic_data_function(x_train, y_train, batch, history, model):
    return x_train, y_train

def basic_lr_scheduler(initial_lr, batch, history):
    return initial_lr


def generate_random_batch(x, y, batch_size):
    size_data = x.shape[0]
    cur_batch_idxs = np.random.choice(size_data, batch_size, replace=False)
    return x[cur_batch_idxs, :, :, :], y[cur_batch_idxs,:]


def train_model_batches(model, dataset, num_batches, batch_size=100,
                        test_each=50, batch_generator=generate_random_batch, initial_lr=1e-3,
                        lr_scheduler=basic_lr_scheduler, loss='categorical_crossentropy',
                        data_function=basic_data_function,
                        verbose=False, target_coverage=None):
    
    x_train = dataset.x_train
    y_train = dataset.y_train_labels
    x_test = dataset.x_test
    y_test = dataset.y_test_labels

    history = {"val_batch_num": [], "data_size": []}
    start_time = time.time()
    for batch in range(num_batches):
        if target_coverage is None: 
            cur_x, cur_y = data_function(x_train, y_train, batch, history, model)
            batch_x, batch_y = batch_generator(cur_x, cur_y, batch_size)
        else:
            cur_easy_x, cur_easy_y, cur_hard_x, cur_hard_y = data_function(x_train, y_train, batch, history, model, target_coverage=target_coverage)
            easy_batch_size = int(target_coverage*batch_size)
            hard_batch_size = batch_size - easy_batch_size
            #print("easy_batch_size: {}, hard_batch_size: {}".format(easy_batch_size, hard_batch_size))
            batch_easy_x, batch_easy_y = batch_generator(cur_easy_x, cur_easy_y, easy_batch_size)
            batch_hard_x, batch_hard_y = batch_generator(cur_hard_x, cur_hard_y, hard_batch_size)
            batch_x = np.concatenate((batch_easy_x, batch_hard_x), axis=0)
            batch_y = np.concatenate((batch_easy_y, batch_hard_y), axis=0)

        cur_lr = lr_scheduler(initial_lr, batch, history)
        K.set_value(model.optimizer.lr, cur_lr)
        #print("y_train shape: {}, batch_y shape: {}".format(y_train.shape, batch_y.shape))
        #cur_loss, cur_accuracy = model.train_on_batch(batch_x, [batch_y, batch_y[:,:-1]])
        metrics = model.train_on_batch(batch_x, [batch_y, batch_y[:,:-1]])
        #print("metrics: {}, metrics_name: {}".format(metrics, model.metrics_names))
        #print("metrics len: {}, metrics name len: {}".format(len(metrics), len(model.metrics_names)))
        for m_idx in range(len(metrics)):
            metric_name = model.metrics_names[m_idx]
            if metric_name not in history:
                history[metric_name] = []
            history[metric_name].append( metrics[m_idx] )
        if target_coverage is None:
            data_size = cur_x.shape[0]
        else:
            data_size = cur_easy_x.shape[0] + cur_hard_x.shape[0]
        history["data_size"].append(data_size)

        if test_each is not None and (batch+1) % test_each == 0:
            #cur_val_loss, cur_val_acc = model.evaluate(x_test, y_test, verbose=0)
            val_metrics = model.evaluate(x_test, [y_test, y_test[:,:-1]], verbose=0)
            #print("val metrics: {}, metrics_name: {}".format(val_metrics, model.metrics_names))
            #print("val metrics len: {}, metrics name len: {}".format(len(val_metrics), len(model.metrics_names)))
            for m_idx in range(len(metrics)):
                metric_name = "val_" + model.metrics_names[m_idx]
                if metric_name not in history:
                    history[metric_name] = []
                history[metric_name].append( metrics[m_idx] )
            #history["val_loss"].append(cur_val_loss)
            #history["val_acc"].append(cur_val_acc)
            history["val_batch_num"].append(batch)
            print("val loss: {}".format(history["val_loss"][-1]))
            #if verbose:
            #    print("val accuracy:", cur_val_acc)
        if verbose and (batch+1) % 10 == 0:
            print("batch: " + str(batch+1) + r"/" + str(num_batches))
            print("last lr used: " + str(cur_lr))
            print("data_size: " + str(data_size))
            print("loss: " + str(history["loss"][-1]))
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()

    return history