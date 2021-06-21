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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(np.mean(y_train), np.mean(y_test))

neighbor_size = 5

stategies = ["distance", "uniform"]

for strategy in stategies:
    print("======================================")
    print("weight strategy: {}".format(strategy))
    neigh = KNeighborsClassifier(n_neighbors=neighbor_size, weights=strategy)
    neigh.fit(X_train, y_train)
    confidence = neigh.predict_proba(X_test)
    max_confidence = np.max(confidence, axis=1)
    prediction = np.argmax(confidence, axis=1)

    #print(max_confidence)
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

        print("acc: {}".format(accuracy))
        print("f1: {}".format(f1))
        print("sensitivity: {}".format(tp/(tp+fn)))
        print("specificity: {}".format(tn/(tn+fp)))
        print("complexity: {}".format(complexity))
        print("")
