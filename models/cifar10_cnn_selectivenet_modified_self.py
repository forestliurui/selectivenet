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
from sklearn.model_selection import train_test_split
from sklearn.covariance import MinCovDet

from selectivnet_utils import *
import cifar10
import cifar100

class cifar10cnn_modi_self:
    def __init__(self, train=True, filename="weightsvgg.h5", coverage=0.8, alpha=0.5, baseline=False, logfile="training.log", datapath=None, target_head=False, **kwargs):
        self.target_coverage = coverage
        if "dataset" in kwargs:
            self.dataset_name = kwargs["dataset"]
            if self.dataset_name == "cifar10":
                self.num_classes = 10
            elif self.dataset_name == "cifar100":
                self.num_classes = 100
        else:
            self.num_classes = 10
        self.alpha = alpha
        if "beta" in kwargs:
            self.beta = kwargs["beta"]
        else:
            self.beta = 1
        if "lamda" in kwargs:
            self.lamda = kwargs["lamda"]
        else:
            self.lamda = 32
        self.logfile = logfile
        self.datapath = datapath
        self.target_head = target_head # false if not want to use the target head to learn the target coverage 
        self.mc_dropout_rate = K.variable(value=0)
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
      
        print("x_train: {}".format(x_train))
        print("x_test: {}".format(x_test))
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
       
        train_size = x_train.shape[0]
        test_size = x_test.shape[0]
        #print("confidence_train: {}".format(confidence_train))
        #print("confidence_test: {}".format(confidence_test))
        confidence_train_portion = int(self.target_coverage*train_size)
        confidence_test_portion = int(self.target_coverage*test_size)
        confidence_train_thresh = np.partition(confidence_train, -confidence_train_portion)[-confidence_train_portion] 
        confidence_test_thresh = np.partition(confidence_test, -confidence_test_portion)[-confidence_test_portion] 
     
        #print("confidence_train_thresh: {}".format(confidence_train_thresh))
        #print("confidence_test_thresh: {}".format(confidence_test_thresh))

        con_label_train = confidence_train > confidence_train_thresh
        con_label_test = confidence_test > confidence_test_thresh
        return con_label_train, con_label_test

    def _load_data(self):
        # The data, shuffled and split between train and test sets:
        if self.dataset_name == "cifar10":
            load_data = cifar10.load_data
        elif self.dataset_name == "cifar100":
            load_data = cifar100.load_data

        (x_train, y_train), (x_test, y_test) = load_data(datapath=self.datapath)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=10000, stratify=y_train)

        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test

        self.y_train = keras.utils.to_categorical(y_train, self.num_classes + 1)
        self.y_val = keras.utils.to_categorical(y_val, self.num_classes)
        self.y_test = keras.utils.to_categorical(y_test, self.num_classes + 1)

        y_train_coverage, y_test_coverage = self._get_confidence_label(
                                                self.x_train, self.y_train, self.x_test, self.y_test, strategy="self")
        
        print("y_train_coverage mean: {}, y_test_coverage_mean: {}".format(np.mean(y_train_coverage), np.mean(y_test_coverage)))
        self.y_train[:,-1] = y_train_coverage
        self.y_test[:,-1] = y_test_coverage

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
                      optimizer=sgd, metrics=['accuracy', selective_acc, coverage])
      
        for epoch in range(maxepoches):
            print("epoch: {}, current lr: {}, iterations: {}".format(epoch, K.get_value(model.optimizer.lr), K.get_value(model.optimizer.iterations)))
            historytemp = model.fit_generator(my_generator(datagen.flow, self.x_train, self.y_train,
                                                           batch_size=batch_size, k=self.num_classes),
                                              steps_per_epoch=self.x_train.shape[0] // batch_size,
                                              epochs=epoch+1, initial_epoch=epoch, callbacks=[reduce_lr, csv_logger],
                                              validation_data=(self.x_test, [self.y_test, self.y_test[:, :-1]]))
            y_train_coverage, y_test_coverage = self._get_confidence_label(
                                                    self.x_train, self.y_train, self.x_test, self.y_test, strategy="self")
            print("y_train_coverage mean: {}, y_test_coverage_mean: {}".format(np.mean(y_train_coverage), np.mean(y_test_coverage)))
            #y_train_coverage = y_train_coverage.reshape((-1,1))
            #y_test_coverage = y_test_coverage.reshape((-1,1))
            self.y_train[:,-1] = y_train_coverage
            self.y_test[:,-1] = y_test_coverage

        #with open("checkpoints/{}_history.pkl".format(self.filename[:-3]), 'wb') as handle:
        #    pickle.dump(historytemp.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #model.save_weights("checkpoints/{}".format(self.filename))
        print("training finished!")

        return model
