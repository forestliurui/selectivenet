import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

data_path = "/home/ruixliu/data/pods/data.csv"
label_path = "/home/ruixliu/data/pods/labels.csv"
# index starting from 0
features_ids = np.array([1,2,3,4,5,6,7,8,9], dtype=np.int32)

label_idx = 10
X = []
y = []

with open(data_path) as csvfile:
    data_reader = csv.reader(csvfile)
    row_idx = -1
    for row in data_reader:
        row_idx += 1
        if row_idx == 0:
            continue
        #X.append(row[np.array(features_ids, dtype=np.int32)])
        feature = []
        for idx, val in enumerate(row):
            if idx >=1:
                try:
                    feature.append(float(row[idx])) 
                except:
                    feature.append(0)
        X.append(feature)
    
with open(label_path) as csvfile:
    label_reader = csv.reader(csvfile)
    row_idx = -1
    for row in label_reader:
        row_idx += 1
        if row_idx == 0:
            continue
        if "AD" in row[1]:
            label = 0
        else:
            label = 1
        y.append(label) 
        #if row_idx < 10: 
        #    print("|".join(row))

X = np.array(X)
y = np.array(y)

print(X.shape)
#print(X)
#print(y)

from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#estimator = SVC(kernel="linear")
#selector = RFE(estimator, n_features_to_select=1000, step=1, verbose=1)
#selector = selector.fit(X, y)
#mask = selector.support_
#print("feature selection mask: {}".format(mask))
#X = X[:,mask]

X = SelectKBest(chi2, k=200).fit_transform(X, y)
print("X size after FS: {}".format(X.shape))

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def compute_metrics(label, prediction):
    accuracy = np.mean(prediction == label)
    f1 = f1_score(label, prediction)
    tn, fp, fn, tp = confusion_matrix(label, prediction).ravel()

    metrics = {}
    metrics["acc"] = accuracy
    metrics["f1"] = f1
    metrics["sensitivity"] = tp/(tp+fn)
    metrics["specificity"] = tn/(tn+fp)

    return metrics 

def train(X_train, y_train, X_test, y_test):
    neighbor_size = 5

    #stategies = ["distance", "uniform"]
    stategies = ["distance"]
    metrics = {}

    for strategy in stategies:
        print("======================================")
        print("weight strategy: {}".format(strategy))
        neigh = KNeighborsClassifier(n_neighbors=neighbor_size, weights=strategy)
        neigh.fit(X_train, y_train)
        confidence = neigh.predict_proba(X_test)
        max_confidence = np.max(confidence, axis=1)
        prediction = np.argmax(confidence, axis=1)

        confidence_train = neigh.predict_proba(X_train)
        prediction_train = np.argmax(confidence_train, axis=1)
        metrics_train = compute_metrics(y_train, prediction_train)
        metrics["train"] = metrics_train

        for thresh in [0.5, 0.7, 0.9, 0.95]:
            print("threshold: {}".format(thresh))
            complexity = 0
            for idx, val in enumerate(max_confidence):
                if val < thresh:
                    complexity += 1
                    prediction[idx] = y_test[idx]
            accuracy = np.mean(prediction == y_test)
            f1 = f1_score(y_test, prediction)
            complexity = complexity/len(prediction)
            tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()

            metrics_test = {}
            metrics_test["acc"] = accuracy
            metrics_test["f1"] = f1
            metrics_test["sensitivity"] = tp/(tp+fn)
            metrics_test["specificity"] = tn/(tn+fp)
            metrics_test["complexity"] = complexity
            metrics["test_threh_"+str(thresh)] = metrics_test

        return metrics

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#print(np.mean(y_train), np.mean(y_test))

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(X, y)
fold_idx = 0
metrics_all = {}
for train_index, test_index in skf.split(X, y):
    print("fold_idx: {}".format(fold_idx))
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("X_train shape: {}; X_test shape: {}".format(X_train.shape, X_test.shape))
    print("train/test positive percent: {:.4f}/{:.4f}".format(np.mean(y_train), np.mean(y_test)))
    

    metrics_all[fold_idx] = train(X_train, y_train, X_test, y_test)

    fold_idx += 1


for method in metrics_all[0].keys():
    print("method: {}".format(method))
    metrics_flat = {"acc": [], 
                    "f1": [], 
                    "sensitivity": [], 
                    "specificity": []}
    for metric_name in metrics_flat.keys():
        for key in metrics_all.keys():
            metrics_flat[metric_name].append(metrics_all[key][method][metric_name])

        print("{}: {}".format(metric_name, metrics_flat[metric_name]))
        print("{}: mean {} std {}".format(metric_name, np.mean(metrics_flat[metric_name]), np.std(metrics_flat[metric_name])))


