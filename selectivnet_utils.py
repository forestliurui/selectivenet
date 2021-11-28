import json
import numpy as np
import os
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from keras.layers import Layer
from keras import backend as K

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
#         print(input_shape)
#         print(self.units)
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

def print_to_log(logfile, msg):
    with open(logfile, "a") as f:
        f.write(msg)
        f.write("\n")


def to_train(filename):
    checkpoints = os.listdir("checkpoints/")
    if filename in checkpoints:
        return False
    else:
        return True


def save_dict(filename, dict):

    with open(filename, 'w') as fp:
        json.dump(dict, fp)


def calc_selective_risk(model, regression, calibrated_coverage=None):
    prediction, pred = model.predict()
    if calibrated_coverage is None:
        threshold = 0.5
    else:
        threshold = np.percentile(prediction[:, -1], 100 - 100 * calibrated_coverage)
    covered_idx = prediction[:, -1] > threshold

    coverage = np.mean(covered_idx)
    y_hat = np.argmax(prediction[:, :-1], 1)
    if regression:
        loss = np.sum(np.mean((prediction[covered_idx, :-1] - model.y_test[covered_idx, :-1]) ** 2, -1)) / np.sum(
            covered_idx)
    else:
        loss = np.sum(y_hat[covered_idx] != np.argmax(model.y_test[covered_idx, :], 1)) / np.sum(covered_idx)
    return loss, coverage


def train_profile(exp_name, model_cls, coverages, dataset=None, model_baseline=None, baseline_name="none", regression=False, alpha=0.5, beta=1, lamda=32, random_percent=-1, random_strategy='feature', order_strategy="inception", logfile='training.log', datapath=None, args=None):
    results = {}
    for coverage_rate in coverages:
        print("running {}_{}.h5".format(exp_name, coverage_rate))
        if model_baseline is None:
            model = model_cls(train=True,
                              filename="{}_{}.h5".format(exp_name, coverage_rate),
                              coverage=coverage_rate,
                              dataset=dataset,
                              alpha=alpha,
                              beta=beta,
                              lamda = lamda,
                              random_percent = random_percent,
                              random_strategy = random_strategy,
                              order_strategy = order_strategy,
                              logfile=logfile,
                              datapath=datapath,
                              args=args
                              )

            loss, coverage = calc_selective_risk(model, regression)
            loss_cali, coverage_cali = calc_selective_risk(model, regression, calibrated_coverage=coverage_rate)

            results[coverage] = {"lambda": coverage_rate, "selective_risk": loss, "selective_risk_calibrated": loss_cali}
        else:
            results[coverage_rate] = {}
        if model_baseline is not None:
            if baseline_name == "mc":
                mc = True
            else:
                mc = False
                
            if regression:
                results[coverage_rate]["baseline_risk"] = (model_baseline.selective_risk_at_coverage(coverage_rate, mc=mc))

            else:
                results[coverage_rate]["baseline_risk"] = (1 - model_baseline.selective_risk_at_coverage(coverage_rate, mc=mc))
            #results[coverage_rate]["percentage"] = 1 - results[coverage_rate]["selective_risk"] / results[coverage_rate]["baseline_risk"]
        print("results: {}".format(results))
        save_dict("results/{}.json".format(exp_name), results)


def post_calibration(exp_name, model_cls, lamda, calibrated_coverage=None, model_baseline=None, regression=False):
    results = {}
    print("calibrating {}_{}.h5".format(exp_name, lamda))
    model = model_cls(train=to_train("{}_{}.h5".format(exp_name, lamda)),
                      filename="{}_{}.h5".format(exp_name, lamda), coverage=lamda)
    loss, coverage = calc_selective_risk(model, regression, calibrated_coverage)

    results[coverage]={"lambda":lamda, "selective_risk":loss}
    if model_baseline is not None:
        if regression:
            results[coverage]["baseline_risk"] = (model_baseline.selective_risk_at_coverage(coverage))

        else:
            results[coverage]["baseline_risk"] = (1 - model_baseline.selective_risk_at_coverage(coverage))
            results[coverage]["mc_risk"] = (1 - model_baseline.selective_risk_at_coverage(coverage, mc=True))

        results[coverage]["percentage"] = 1 - results[coverage]["selective_risk"] / results[coverage]["baseline_risk"]

    return results


def my_generator(func, x_train, y_train, batch_size, k=10):
    while True:
        res = func(x_train, y_train, batch_size
                   ).next()
        yield [res[0], [res[1], res[1][:, :-1]]]


def create_cats_vs_dogs_npz(cats_vs_dogs_path='datasets'):
    labels = ['cat', 'dog']
    label_to_y_dict = {l: i for i, l in enumerate(labels)}

    def _load_from_dir(dir_name):
        glob_path = os.path.join(cats_vs_dogs_path, dir_name, '*.*.jpg')
        imgs_paths = glob(glob_path)
        images = [resize_and_crop_image(p, 64) for p in imgs_paths]
        x = np.stack(images)
        y = [label_to_y_dict[os.path.split(p)[-1][:3]] for p in imgs_paths]
        y = np.array(y)
        return x, y

    x_train, y_train = _load_from_dir('train')
    x_test, y_test = _load_from_dir('test')

    np.savez_compressed(os.path.join(cats_vs_dogs_path, 'cats_vs_dogs.npz'),
                        x_train=x_train, y_train=y_train,
                        x_test=x_test, y_test=y_test)


def load_cats_vs_dogs(cats_vs_dogs_path='datasets/'):
    npz_file = np.load(os.path.join(cats_vs_dogs_path, 'cats_vs_dogs.npz'))
    x_train = npz_file['x_train']
    y_train = npz_file['y_train']
    x_test = npz_file['x_test']
    y_test = npz_file['y_test']

    return (x_train, y_train), (x_test, y_test)
