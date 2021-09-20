from __future__ import print_function

import keras
import numpy as np
import pickle
import scipy.io as spio
from argparse import Namespace
from keras import backend as K
from keras import backend as K
from keras import optimizers
from keras import regularizers
from keras.datasets import cifar10
from keras.engine.topology import Layer
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Concatenate
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers.core import Lambda
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.covariance import MinCovDet
from sklearn.model_selection import train_test_split

from selectivnet_utils import *
from curriculum_learning.datasets.svhn import SVHN
#import cifar10
#import cifar100

class Svhncnn_curr:
    def __init__(self, train=True, filename="weightsvgg.h5", coverage=0.8, alpha=0.5, baseline=False, logfile="training.log", datapath=None, **kwargs):
        self.target_coverage = coverage
        self.alpha = alpha
        self.logfile = logfile
        self.datapath = datapath
        self.mc_dropout_rate = K.variable(value=0)
        self.dataset_name = "svhn"
        self.num_classes = 10
        self.weight_decay = 0.0005
        if "lamda" in kwargs:
            self.lamda = kwargs["lamda"]
        else:
            self.lamda = 32
        if "random_percent" in kwargs:
            self.random_percent = kwargs["random_percent"]
        else:
            self.random_percent = -1
        if "random_strategy" in kwargs:
            self.random_strategy = kwargs["random_strategy"]
        else:
            self.random_strategy = "feature"
        if "order_strategy" in kwargs:
            self.order_strategy = kwargs["order_strategy"]
        else:
            self.order_strategy = "inception"
        if "maxepoches" in kwargs:
            self.maxepoches = kwargs["maxepoches"]
        else:
            self.maxepoches = None
        if "input_data" in kwargs:
            self.input_data = kwargs["input_data"]
        else:
            self.input_data = None
        if "args" in kwargs:
            self.args = kwargs["args"]
        else:
            self.args = None
        if self.args is not None:
            self.curriculum = getattr(self.args, "curriculum_strategy", "curriculum")
        else:
            self.curriculum = "curriculum"
        print("curriculum strategy: {}".format(self.curriculum))
        print("model args: {}".format(kwargs))

        if self.input_data is None:
            print("use loaded data")
            self._load_data()
        else:
            print("use input data from argument")
            self.x_train = self.input_data["x_train"] 
            self.y_train = self.input_data["y_train"] 
            self.x_test = self.input_data["x_test"] 
            self.y_test = self.input_data["y_test"]

        self.x_shape = self.x_train.shape[1:]
        self.filename = filename

        self.model = self.build_model()
        if baseline:
            self.alpha = 0

        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights("checkpoints/{}".format(self.filename))

    def build_model(self, self_taught=False):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
        weight_decay = self.weight_decay
        basic_dropout_rate = 0.3
        input_shape = self.x_shape
        batch_norm = True
        l2_reg = regularizers.l2(weight_decay) #K.variable(K.cast_to_floatx(reg_factor))
        l2_bias_reg = None
        #if bias_reg_factor:
        #    l2_bias_reg = regularizers.l2(bias_reg_factor) #K.variable(K.cast_to_floatx(bias_reg_factor))
        dropout_1_rate = 0.25
        dropout_2_rate = 0.5
        activation = "elu"

        x = input = Input(shape=input_shape)
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(rate=dropout_1_rate)(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(rate=dropout_1_rate)(x)

        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(rate=dropout_1_rate)(x)

        x = Conv2D(filters=256, kernel_size=(2, 2), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        x = Conv2D(filters=256, kernel_size=(2, 2), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(rate=dropout_1_rate)(x)

        x = Flatten()(x)
        x = Dense(units=512, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)

        curr = Dropout(rate=dropout_2_rate, name='feature_layer')(x)

        # classification head (f)
        curr1 = Dense(self.num_classes, activation='softmax')(curr)

        # selection head (g)
        curr2 = Dense(512, kernel_regularizer=regularizers.l2(weight_decay))(curr)
        curr2 = Activation('relu')(curr2)
        curr2 = BatchNormalization()(curr2)
        # this normalization is identical to initialization of batchnorm gamma to 1/10
        curr2 = Lambda(lambda x: x / 10)(curr2)
        curr2 = Dense(1, activation='sigmoid')(curr2)
        # auxiliary head (h)
        selective_output = Concatenate(axis=1, name="selective_head")([curr1, curr2])

        auxiliary_output = Dense(self.num_classes, activation='softmax', name="classification_head")(curr)

        if self_taught is False:
            model = Model(inputs=input, outputs=[selective_output, auxiliary_output])
        else:
            model = Model(inputs=input, outputs=auxiliary_output)

        self.input = input
        self.model_embeding = Model(inputs=input, outputs=curr)
        return model

    def normalize(self, X_train, X_test, axis=(0,1,2,3)):
        # this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train, axis=axis)
        std = np.std(X_train, axis=axis)
        X_train = (X_train - mean) / (std + 1e-7)
        X_test = (X_test - mean) / (std + 1e-7)
        return X_train, X_test

    def predict(self, x=None, batch_size=128):
        if x is None:
            x = self.x_test
        return self.model.predict(x, batch_size)

    def predict_embedding(self, x=None, batch_size=128):
        if x is None:
            x = self.x_test
        return self.model_embeding.predict(x, batch_size)

    def mc_dropout(self, batch_size=1000, dropout=0.5, iter=100):
        K.set_value(self.mc_dropout_rate, dropout)
        repititions = []
        for i in range(iter):
            _, pred = self.model.predict(self.x_test, batch_size)
            repititions.append(pred)
        K.set_value(self.mc_dropout_rate, 0)

        repititions = np.array(repititions)
        mc = np.var(repititions, 0)
        mc = np.mean(mc, -1)
        return -mc

    def selective_risk_at_coverage(self, coverage, mc=False):
        _, pred = self.predict()

        if mc:
            sr = np.max(pred, 1)
        else:
            sr = self.mc_dropout()
        sr_sorted = np.sort(sr)
        threshold = sr_sorted[pred.shape[0] - int(coverage * pred.shape[0])]
        covered_idx = sr > threshold
        selective_acc = np.mean(np.argmax(pred[covered_idx], 1) == np.argmax(self.y_test[covered_idx], 1))
        return selective_acc

    def randomize_label(self, x_train, y_train, x_test, y_test):
        print("run randomize_label")
        num_train = x_train.shape[0]
        num_test = x_test.shape[0]
        random_idx_train = np.unique(np.random.randint(num_train, size=int(num_train*self.random_percent/100))) 
        random_idx_test = np.unique(np.random.randint(num_test, size=int(num_test*self.random_percent/100))) 

        y_train_flatten = np.argmax(y_train, axis=1)
        y_test_flatten = np.argmax(y_test, axis=1)
        
        y_train_random = np.random.randint(self.num_classes, size=len(random_idx_train))
        y_test_random = np.random.randint(self.num_classes, size=len(random_idx_test))

        y_train_flatten[random_idx_train] = y_train_random
        y_test_flatten[random_idx_test] = y_test_random

        self.y_train = keras.utils.to_categorical(y_train_flatten, self.num_classes + 1)
        self.y_test = keras.utils.to_categorical(y_test_flatten, self.num_classes + 1)

        con_label_train = np.ones(num_train)
        con_label_test = np.ones(num_test)

        con_label_train[random_idx_train] = 0
        con_label_test[random_idx_test] = 0

        print("y_train_coverage mean: {}, y_test_coverage_mean: {}".format(np.mean(con_label_train), np.mean(con_label_test)))
        return con_label_train, con_label_test

    def randomize_feature(self, x_train, y_train, x_test, y_test):
        print("run randomize_feature")
        num_train = x_train.shape[0]
        num_test = x_test.shape[0]
        random_idx_train = np.unique(np.random.randint(num_train, size=int(num_train*self.random_percent/100))) 
        random_idx_test = np.unique(np.random.randint(num_test, size=int(num_test*self.random_percent/100))) 

        for r_idx in random_idx_train:
            self.x_train[r_idx,:] = np.random.permutation(self.x_train[r_idx,:])

        for r_idx in random_idx_test:
            self.x_test[r_idx,:] = np.random.permutation(self.x_test[r_idx,:])

        con_label_train = np.ones(num_train)
        con_label_test = np.ones(num_test)

        con_label_train[random_idx_train] = 0
        con_label_test[random_idx_test] = 0

        print("y_train_coverage mean: {}, y_test_coverage_mean: {}".format(np.mean(con_label_train), np.mean(con_label_test)))
        return con_label_train, con_label_test

    def randomize_feature_gaussian(self, x_train, y_train, x_test, y_test):
        print("run randomize_feature_gaussian")
        num_train = x_train.shape[0]
        num_test = x_test.shape[0]
        random_idx_train = np.unique(np.random.randint(num_train, size=int(num_train*self.random_percent/100))) 
        random_idx_test = np.unique(np.random.randint(num_test, size=int(num_test*self.random_percent/100))) 

        for r_idx in random_idx_train:
            mean = np.mean(self.x_train[r_idx,:])
            std = np.std(self.x_train[r_idx,:])
            shape = self.x_train[r_idx,:].shape
            self.x_train[r_idx,:] += np.random.normal(mean, std, shape)

        for r_idx in random_idx_test:
            mean = np.mean(self.x_test[r_idx,:])
            std = np.std(self.x_test[r_idx,:])
            shape = self.x_test[r_idx,:].shape
            self.x_test[r_idx,:] += np.random.normal(mean, std, shape)

        con_label_train = np.ones(num_train)
        con_label_test = np.ones(num_test)

        con_label_train[random_idx_train] = 0
        con_label_test[random_idx_test] = 0

        print("y_train_coverage mean: {}, y_test_coverage_mean: {}".format(np.mean(con_label_train), np.mean(con_label_test)))
        return con_label_train, con_label_test

    def _get_order(self, x_train, y_train, x_test, y_test, strategy="self"):
        """
        The confidence label of an example is 1 if we have high confidence on this example, and 
        the model should give normal prediction on this example;
        otherwise, the confidence label is 0
        """
      
        #print("x_train: {}".format(x_train))
        #print("x_test: {}".format(x_test))
        if strategy == "self":
            self.x_shape = x_train.shape[1:]
            confidence_model = self.build_model(self_taught=True)
            confidence_model.compile(optimizer='sgd',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            print("self.x_val (type: {}): {}".format(type(self.x_val), self.x_val))
            print("self.y_val (type: {}): {}".format(type(self.y_val), self.y_val))
            confidence_model.fit(self.x_val, self.y_val, epochs=150)
            loss_train = confidence_model.predict(x=x_train)[1]
            loss_test = confidence_model.predict(x=x_test)[1]

            feature_extractor = Model(inputs=confidence_model.input, outputs=confidence_model.get_layer("feature_layer").output)
            x_train_trans = feature_extractor.predict(x=x_train)
            x_test_trans = feature_extractor.predict(x=x_test)
            print("x_train_trans shape: {}, x_test_trans shape: {}".format(x_train_trans.shape, x_test_trans.shape ))
            x_train_trans, x_test_trans = self.normalize(x_train_trans, x_test_trans, axis=0)
            print("x_train_trans and x_test_trans normalized!")
            print("x_train_trans: {}".format(x_train_trans))
            print("x_test_trans: {}".format(x_test_trans))

            confidence_train = np.zeros(x_train.shape[0])
            for class_idx in range(self.num_classes):
                train_idx = (np.argmax(y_train, axis=1) == class_idx)
                feature_per_class = x_train_trans[train_idx]
                num_example = feature_per_class.shape[0]
                feature_per_class = feature_per_class.reshape(num_example, -1)
                conv_per_class = MinCovDet(random_state=0).fit(feature_per_class)
                m_distance = conv_per_class.mahalanobis(feature_per_class)

                confidence_train[train_idx] = -m_distance
                print("train class idx: {}, m_distance: {}".format(class_idx, m_distance))

            confidence_test = np.zeros(x_test.shape[0])
            for class_idx in range(self.num_classes):
                test_idx = (np.argmax(y_test, axis=1) == class_idx)
                feature_per_class = x_test_trans[test_idx]
                num_example = feature_per_class.shape[0]
                feature_per_class = feature_per_class.reshape(num_example, -1)
                conv_per_class = MinCovDet(random_state=0).fit(feature_per_class)
                m_distance = conv_per_class.mahalanobis(feature_per_class)

                confidence_test[test_idx] = -m_distance
                print("test class idx: {}, m_distance: {}".format(class_idx, m_distance))

            order = np.asarray(sorted(range(len(confidence_train)), key=lambda k: confidence_train[k], reverse=True))

            return order

    def _load_data(self):
        # The data, shuffled and split between train and test sets:
        #if self.dataset == "cifar10":
        #    load_data = cifar10.load_data
        #elif self.dataset == "cifar100":
        #    load_data = cifar100.load_data
        


        self.dataset = SVHN(data_path=self.datapath, normalize=False) 
        x_train = self.dataset.x_train
        y_train = self.dataset.y_train_labels
        x_test = self.dataset.x_test
        y_test = self.dataset.y_test_labels
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)
        
        if self.order_strategy == "self_split":
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=10000, stratify=y_train)
        else:
            x_train = np.copy(x_train)
            x_val = np.copy(x_train)
            y_train = np.copy(y_train)
            y_val = np.copy(y_train)

        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        
        print("x_train shape: {}, x_val shape: {}, x_test shape: {}".format(x_train.shape, x_val.shape, x_test.shape))

        y_train = np.argmax(y_train, 1)
        y_test = np.argmax(y_test, 1)
        y_val = np.argmax(y_val, 1)
        self.y_train = keras.utils.to_categorical(y_train, self.num_classes + 1)
        self.y_val = keras.utils.to_categorical(y_val, self.num_classes)
        self.y_test = keras.utils.to_categorical(y_test, self.num_classes + 1)

        if self.random_percent > 0:
            if self.random_strategy == "label":
                randomize_fn = self.randomize_label
            elif self.random_strategy == "feature":
                randomize_fn = self.randomize_feature
            elif self.random_strategy == "feature_gaussian":
                randomize_fn = self.randomize_feature_gaussian
            else:
                raise ValueError("random strategy not supported: {}".format(self.random_strategy))
            y_train_coverage, y_test_coverage = randomize_fn(self.x_train, self.y_train, self.x_test, self.y_test)

            self.y_train[:,-1] = y_train_coverage
            self.y_test[:,-1] = y_test_coverage
        
        self.dataset.x_train = self.x_train
        self.dataset.y_train_labels = self.y_train
        self.dataset.x_test = self.x_test
        self.dataset.y_test_labels = self.y_test
    
    def train(self, model):
        from curriculum_learning.main_train_networks import load_order, balance_order
        #from ..curriculum_learning import main_train_networks.load_order
        #import curriculum_learning.main_train_networks.load_order
        print("import curriculum learning module!!!")
        if self.order_strategy == "inception":
            order = load_order("inception", self.dataset)
            print("order: {}".format(order[:100]))
            order = balance_order(order, self.dataset)
            print("new order: {}".format(order[:100]))
        elif self.order_strategy in ["self", "self_split"]:
            order = self._get_order(self.x_train, self.y_train, self.x_test, self.y_test)
            print("order: {}".format(order[:100]))
            order = balance_order(order, self.dataset)
            print("new order: {}".format(order[:100]))
        else:
            order=None
      
        if self.curriculum == "anti":
            order = np.flip(order, 0)
        elif self.curriculum == "random":
            np.random.shuffle(order)
        
        c = self.target_coverage
        def selective_loss(y_true, y_pred):
            loss = K.categorical_crossentropy(
                K.repeat_elements(y_pred[:, -1:], self.num_classes, axis=1) * y_true[:, :-1],
                y_pred[:, :-1]) + self.lamda * K.maximum(-K.mean(y_pred[:, -1]) + c, 0) ** 2
            return loss

        def selective_acc(y_true, y_pred):
            g = K.cast(K.greater(y_pred[:, -1], 0.5), K.floatx())
            temp1 = K.sum(
                (g) * K.cast(K.equal(K.argmax(y_true[:, :-1], axis=-1), K.argmax(y_pred[:, :-1], axis=-1)), K.floatx()))
            temp1 = temp1 / K.sum(g)
            return K.cast(temp1, K.floatx())

        def coverage(y_true, y_pred):
            g = K.cast(K.greater(y_pred[:, -1], 0.5), K.floatx())
            return K.mean(g)

        def confidence_acc(y_true, y_pred):
            g_pred = K.cast(K.greater(y_pred[:, -1], 0.5), K.floatx())
            g_true = K.cast(y_true[:, -1], K.floatx())
            temp1 = K.cast(K.equal(g_true, g_pred), K.floatx())
            return K.mean(temp1)

        def confidence_loss(y_true, y_pred):
            c_loss = K.binary_crossentropy(y_true[:,-1], y_pred[:,-1])
            return c_loss

        # training parameters
        batch_size = 128
        if self.maxepoches is None:
            maxepoches = 300
        else:
            maxepoches = self.maxepoches
        num_batches = (self.dataset.x_train.shape[0] * maxepoches) // batch_size

        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 25

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))

        def lr_scheduler_iter(initial_lr, batch, history):
            num_batch_per_epoch = self.dataset.x_train.shape[0]//batch_size
            epoch = batch//num_batch_per_epoch
            return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
        csv_logger = keras.callbacks.CSVLogger(self.logfile, append=True)

        from curriculum_learning.main_train_networks import data_function_from_input
        from curriculum_learning import train_keras_model
      
        self.curriculum_args = Namespace(dataset="svhn",
                         model=model,
                         verbose=True,
                         optimizer="sgd",
                         curriculum="vanilla",
                         batch_size=100,
                         num_epochs=140,
                         learning_rate=0.12,
                         lr_decay_rate=1.1,
                         minimal_lr=1e-3,
                         lr_batch_size=700,
                         batch_increase=100,
                         increase_amount=1.9,
                         starting_percent=0.04,
                         order="inception",
                         test_each=50,
                         balance=True)
        data_function = data_function_from_input(self.curriculum,
                                                 batch_size,
                                                 self.dataset,
                                                 order,
                                                 self.curriculum_args.batch_increase,
                                                 self.curriculum_args.increase_amount,
                                                 self.curriculum_args.starting_percent)
        
        if self.curriculum == "partition":
            target_coverage = c
        else:
            target_coverage = None
        print("curriculum strategy: {}, target_coverage if partition: {}".format(self.curriculum, target_coverage))
        # optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)

        model.compile(loss=[selective_loss, 'categorical_crossentropy'], loss_weights=[self.alpha, 1 - self.alpha],
                      optimizer=sgd, metrics=['accuracy', selective_acc, coverage, confidence_loss, confidence_acc])
        history = train_keras_model.train_model_batches(model, 
                                                           self.dataset, 
                                                           num_batches, 
                                                           batch_size=batch_size,
                                                           initial_lr=learning_rate,
                                                           lr_scheduler=lr_scheduler_iter,
                                                           verbose=self.curriculum_args.verbose,
                                                           data_function=data_function,
                                                           target_coverage=target_coverage
                                                           ) 

        # historytemp = model.fit_generator(my_generator(datagen.flow, self.x_train, self.y_train,
        #                                                batch_size=batch_size, k=self.num_classes),
        #                                   steps_per_epoch=self.x_train.shape[0] // batch_size,
        #                                   epochs=maxepoches, callbacks=[reduce_lr, csv_logger],
        #                                  validation_data=(self.x_test, [self.y_test, self.y_test[:, :-1]]))


        #with open("checkpoints/{}_history.pkl".format(self.filename[:-3]), 'wb') as handle:
        #    pickle.dump(historytemp.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #model.save_weights("checkpoints/{}".format(self.filename))
        for metric_name in history:
            print("{}: {}".format(metric_name, history[metric_name]))
        print("training finished!")

        return model
