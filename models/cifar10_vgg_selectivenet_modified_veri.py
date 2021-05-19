from __future__ import print_function

import keras
import numpy as np
import pickle
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

from selectivnet_utils import *
from cifar10 import *

class cifar10vgg_modi_veri:
    def __init__(self, train=True, filename="weightsvgg.h5", coverage=0.8, alpha=0.5, baseline=False, logfile="training.log", datapath=None, target_head=False, **kwargs):
        self.target_coverage = coverage
        self.alpha = alpha
        if "beta" in kwargs:
            self.beta = kwargs["beta"]
        else:
            self.beta = 1
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
        self.logfile = logfile
        self.datapath = datapath
        self.target_head = target_head # false if not want to use the target head to learn the target coverage 
        self.mc_dropout_rate = K.variable(value=0)
        self.num_classes = 10
        self.weight_decay = 0.0005

        print("model args: {}".format(kwargs))

        self._load_data()
        self.x_shape = self.x_train.shape[1:]
        self.filename = filename

        self.model = self.build_model()
        if baseline:
            self.alpha = 0

        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights("checkpoints/{}".format(self.filename))

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
        weight_decay = self.weight_decay
        basic_dropout_rate = 0.3
        input = Input(shape=self.x_shape)
        curr = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(input)
        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate)(curr)

        curr = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2))(curr)

        curr = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2))(curr)

        curr = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2))(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2))(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2))(curr)
        curr = Dropout(basic_dropout_rate + 0.2)(curr)

        curr = Flatten()(curr)
        curr = Dense(512, kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.2)(curr)
        curr = Lambda(lambda x: K.dropout(x, level=self.mc_dropout_rate))(curr)

        # classification head (f)
        curr1 = Dense(self.num_classes, activation='softmax')(curr)

        # selection head (g)
        curr2 = Dense(512, kernel_regularizer=regularizers.l2(weight_decay))(curr)
        curr2 = Activation('relu')(curr2)
        curr2 = BatchNormalization()(curr2)
        # this normalization is identical to initialization of batchnorm gamma to 1/10
        curr2 = Lambda(lambda x: x / 10)(curr2)
        curr2 = Dense(1, activation='sigmoid')(curr2)
        
        selective_output = Concatenate(axis=1, name="selective_head")([curr1, curr2])

        # target head (t)
        if self.target_head is True:
            target_output = Dense(1, activation='sigmoid')(curr)
            self.target_coverage = K.mean(target_output)

        # auxiliary head (h)
        auxiliary_output = Dense(self.num_classes, activation='softmax', name="classification_head")(curr)

        model = Model(inputs=input, outputs=[selective_output, auxiliary_output])

        self.input = input
        self.model_embeding = Model(inputs=input, outputs=curr)
        return model

    def normalize(self, X_train, X_test):
        # this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train, axis=(0, 1, 2, 3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
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

    def _get_confidence_label(self, x_train, y_train, x_test, y_test, strategy="logit"):
        """
        The confidence label of an example is 1 if we have high confidence on this example, and 
        the model should give normal prediction on this example;
        otherwise, the confidence label is 0
        """
      
        if strategy == "logit":
            print("run confidence label strategy: logit")
            x_shape = x_train.shape[1:]
            input = Input(shape=x_shape)
            curr = Flatten()(input) 
            output = Dense(self.num_classes, activation='softmax', kernel_regularizer=regularizers.l2(self.weight_decay))(curr)
            
            confidence_model = Model(inputs=input, outputs=output)
            confidence_model.compile(optimizer='sgd',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            confidence_model.fit(x_train, y_train, epochs=10)
            loss_train = confidence_model.predict(x=x_train)
            loss_test = confidence_model.predict(x=x_test)
        elif strategy == "rbf":
            print("run confidence label strategy: rbf")
            x_shape = x_train.shape[1:]
            input = Input(shape=x_shape)
            curr = Flatten()(input) 
            curr = RBFLayer(10, 0.5)(curr)
            output = Dense(self.num_classes, activation='softmax', kernel_regularizer=regularizers.l2(self.weight_decay))(curr)
           

            confidence_model = Model(inputs=input, outputs=output)
            confidence_model.compile(optimizer='sgd',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            confidence_model.fit(x_train, y_train, epochs=10)
            loss_train = confidence_model.predict(x=x_train)
            loss_test = confidence_model.predict(x=x_test)
        elif strategy == "self":
            print("run confidence label strategy: self")
            confidence_model = self.model
            loss_train = confidence_model.predict(x=x_train)[1]
            loss_test = confidence_model.predict(x=x_test)[1]

        elif strategy == "randomize_label":
            print("run confidence label strategy: randomize_label")
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

            self.y_train = keras.utils.to_categorical(y_train_flatten, self.num_classes)
            self.y_test = keras.utils.to_categorical(y_test_flatten, self.num_classes)

            con_label_train = np.ones(num_train)
            con_label_test = np.ones(num_test)

            con_label_train[random_idx_train] = 0
            con_label_test[random_idx_test] = 0

            return con_label_train, con_label_test 

        elif strategy == "randomize_feature":
            print("run confidence label strategy: randomize_feature")
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

            return con_label_train, con_label_test 

        elif strategy == "randomize_feature_baseline":
            # use baseline model to obtain the confidence labels
            print("run confidence label strategy: randomize_feature_baseline")
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

            y_train_coverage = con_label_train.reshape((-1,1))
            y_test_coverage = con_label_test.reshape((-1,1))
            y_train_baseline = np.concatenate((self.y_train, y_train_coverage), axis=1)
            y_test_baseline = np.concatenate((self.y_test, y_test_coverage), axis=1)

            from .cifar10_vgg_selectivenet import cifar10vgg
            input_data_baseline = {"x_train": self.x_train, 
                                   "y_train": y_train_baseline, 
                                   "x_test": self.x_test,
                                   "y_test": y_test_baseline}

            self.baseline = cifar10vgg(train=True,
                          filename="{}_{}.h5".format("cifar10vgg", self.target_coverage),
                          coverage=self.target_coverage,
                          alpha=self.alpha,
                          beta=self.beta,
                          lamda = self.lamda,
                          random_percent = self.random_percent,
                          random_strategy = self.random_strategy,
                          logfile=self.logfile,
                          datapath=self.datapath,
                          input_data = input_data_baseline, 
                          ) 

            selective_loss_train, aux_loss_train = self.baseline.predict(self.x_train)
            selective_loss_test, aux_loss_test = self.baseline.predict(self.x_test)

            con_label_train_from_baseline = (selective_loss_train[:,-1]>0.5).reshape(-1)
            con_label_test_from_baseline = (selective_loss_test[:,-1]>0.5).reshape(-1)

            return con_label_train_from_baseline, con_label_test_from_baseline 

        elif strategy == "randomize_feature_gaussian":
            print("run confidence label strategy: randomize_feature_gaussian")
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

            return con_label_train, con_label_test 

        elif strategy == "randomize_feature_gaussian_baseline":
            # use baseline model to obtain the confidence labels
            print("run confidence label strategy: randomize_feature_baseline")
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

            y_train_coverage = con_label_train.reshape((-1,1))
            y_test_coverage = con_label_test.reshape((-1,1))
            y_train_baseline = np.concatenate((self.y_train, y_train_coverage), axis=1)
            y_test_baseline = np.concatenate((self.y_test, y_test_coverage), axis=1)

            from .cifar10_vgg_selectivenet import cifar10vgg
            input_data_baseline = {"x_train": self.x_train, 
                                   "y_train": y_train_baseline, 
                                   "x_test": self.x_test,
                                   "y_test": y_test_baseline}

            self.baseline = cifar10vgg(train=True,
                          filename="{}_{}.h5".format("cifar10vgg", self.target_coverage),
                          coverage=self.target_coverage,
                          alpha=self.alpha,
                          beta=self.beta,
                          lamda = self.lamda,
                          random_percent = self.random_percent,
                          random_strategy = self.random_strategy,
                          logfile=self.logfile,
                          datapath=self.datapath,
                          input_data = input_data_baseline, 
                          ) 

            selective_loss_train, aux_loss_train = self.baseline.predict(self.x_train)
            selective_loss_test, aux_loss_test = self.baseline.predict(self.x_test)

            con_label_train_from_baseline = (selective_loss_train[:,-1]>0.5).reshape(-1)
            con_label_test_from_baseline = (selective_loss_test[:,-1]>0.5).reshape(-1)

            return con_label_train_from_baseline, con_label_test_from_baseline 

        confidence_train = np.max(loss_train, 1)
        confidence_test = np.max(loss_test, 1)
        
        con_label_train = confidence_train > 0.3
        con_label_test = confidence_test > 0.3
        return con_label_train, con_label_test

    def _load_data(self):

        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test_label) = load_data(self.datapath)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        self.x_train, self.x_test = self.normalize(x_train, x_test)

        # self.y_train = keras.utils.to_categorical(y_train, self.num_classes + 1)
        # self.y_test = keras.utils.to_categorical(y_test_label, self.num_classes + 1)

        self.y_train = keras.utils.to_categorical(y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(y_test_label, self.num_classes)

        if self.random_strategy == "label":
            strategy = "randomize_label"
        elif self.random_strategy == "feature":
            strategy = "randomize_feature"
        elif self.random_strategy == "feature_baseline":
            strategy = "randomize_feature_baseline"
        elif self.random_strategy == "feature_gaussian":
            strategy = "randomize_feature_gaussian"
        elif self.random_strategy == "feature_gaussian_baseline":
            strategy = "randomize_feature_gaussian_baseline"
        else:
            strategy = "logit"

        y_train_coverage, y_test_coverage = self._get_confidence_label(
                                                self.x_train, self.y_train, self.x_test, self.y_test, strategy=strategy)
        print("y_train_coverage mean: {}, y_test_coverage_mean: {}".format(np.mean(y_train_coverage), np.mean(y_test_coverage)))
        
        #num_train = x_train.shape[0] 
        #num_test = x_test.shape[0]
        #y_train_coverage = np.random.uniform(size=num_train)<self.target_coverage
        #y_test_coverage = np.random.uniform(size=num_test)<self.target_coverage

        y_train_coverage = y_train_coverage.reshape((-1,1))
        y_test_coverage = y_test_coverage.reshape((-1,1))
        self.y_train = np.concatenate((self.y_train, y_train_coverage), axis=1)
        self.y_test = np.concatenate((self.y_test, y_test_coverage), axis=1)

    def train(self, model):
        c = self.target_coverage

        def selective_loss(y_true, y_pred):
            s_loss = K.categorical_crossentropy(
                K.repeat_elements(y_pred[:, -1:], self.num_classes, axis=1) * y_true[:, :-1],
                y_pred[:, :-1]) + self.lamda * K.maximum(-K.mean(y_pred[:, -1]) + c, 0) ** 2
            
            c_loss = K.binary_crossentropy(y_true[:,-1], y_pred[:,-1])
            loss = s_loss + self.beta * c_loss
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

        class CustomCallback(keras.callbacks.Callback):
            def __init__(self):
                super(CustomCallback, self).__init__()
                print(dir(self.model))
                #print(dir(self.model.optimizer))

            def on_train_batch_end(self, batch, logs=None):
                keys = list(logs.keys())
                print("...Training: end of batch {}; got log keys: {}".format(batch, keys))
                #print(dir(self.model))

        # training parameters
        batch_size = 128
        maxepoches = 300
        learning_rate = 0.1

        lr_decay = 1e-6

        lr_drop = 25

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
        csv_logger = keras.callbacks.CSVLogger(self.logfile, append=True)

        # data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(self.x_train)

        # optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)

        model.compile(loss=[selective_loss, 'categorical_crossentropy'], loss_weights=[self.alpha, 1 - self.alpha],
                      optimizer=sgd, metrics=['accuracy', selective_acc, coverage, confidence_loss, confidence_acc])
      
        for epoch in range(maxepoches):
            print("epoch: {}, current lr: {}, iterations: {}".format(epoch, K.get_value(model.optimizer.lr), K.get_value(model.optimizer.iterations)))
            historytemp = model.fit_generator(my_generator(datagen.flow, self.x_train, self.y_train,
                                                           batch_size=batch_size, k=self.num_classes),
                                              steps_per_epoch=self.x_train.shape[0] // batch_size,
                                              epochs=epoch+1, initial_epoch=epoch, callbacks=[reduce_lr, csv_logger],
                                              validation_data=(self.x_test, [self.y_test, self.y_test[:, :-1]]))
            #y_train_coverage, y_test_coverage = self._get_confidence_label(
            #                                        self.x_train, self.y_train, self.x_test, self.y_test, strategy="self")
            #print("y_train_coverage mean: {}, y_test_coverage_mean: {}".format(np.mean(y_train_coverage), np.mean(y_test_coverage)))
            #self.y_train[:,-1] = y_train_coverage
            #self.y_test[:,-1] = y_test_coverage

        #with open("checkpoints/{}_history.pkl".format(self.filename[:-3]), 'wb') as handle:
        #    pickle.dump(historytemp.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #model.save_weights("checkpoints/{}".format(self.filename))
        print("training finished!")

        return model
