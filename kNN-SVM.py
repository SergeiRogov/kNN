import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_wine
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def train_and_test_model(dataset, model):
    # splitting dataset - 70% for training and 30% for testing
    x, y = dataset.data, dataset.target
    # kf = KFold(n_splits=10)
    # for train_index, test_index in kf.split():
    #     cross_val_score(model, x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # training a model
    model = model.fit(x_train, y_train)
    # testing a model with a test set
    predict = model.predict(x_test)
    # Printing out a report on accuracy
    print(f"Accuracy = {accuracy_score(y_test, predict)}")
    print(f"Precision = {precision_score(y_test, predict, average=None)}")
    print(f"Recall = {recall_score(y_test, predict, average=None)}")
    print(f"F-measure = {f1_score(y_test, predict, average=None)}\n")


# loading iris and wine dataset
iris = load_iris()
wine = load_wine()

# creating knn and svm models
knn_model = KNeighborsClassifier(n_neighbors=5)
svm_model = SVC()

# forming lists
datasets = [iris, wine]
models = [knn_model, svm_model]

# for dataset in datasets:
#     for model in models:
#         train_and_test_model(dataset, model)
print("iris, knn_model")
train_and_test_model(iris, knn_model)
print("iris, svm_model")
train_and_test_model(iris, svm_model)
print("wine, knn_model")
train_and_test_model(wine, knn_model)
print("wine, svm_model")
train_and_test_model(wine, svm_model)
