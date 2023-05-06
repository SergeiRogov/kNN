import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_wine
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer


def train_and_test_model(dataset, model):
    x, y = dataset.data, dataset.target
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, average='macro', zero_division=1),
               'recall': make_scorer(recall_score, average='macro'),
               'f1_score': make_scorer(f1_score, average='macro')}
    scores = cross_validate(model, x, y, cv=10, scoring=scoring)

    # print(f"10-Fold Validation accuracy = {scores}\n")
    print(f"Accuracy = {scores['test_accuracy'].mean():.3f}")
    print(f"Precision = {scores['test_precision'].mean():.3f}")
    print(f"Recall = {scores['test_recall'].mean():.3f}")
    print(f"F_score = {scores['test_f1_score'].mean():.3f}\n")


# loading iris and wine dataset
iris = load_iris()
wine = load_wine()

# creating knn and svm models
knn_model = KNeighborsClassifier(n_neighbors=5)
svm_model = SVC()

print("iris, kNN")
train_and_test_model(iris, knn_model)
print("iris, SVM")
train_and_test_model(iris, svm_model)
print("wine, kNN")
train_and_test_model(wine, knn_model)
print("wine, SVM")
train_and_test_model(wine, svm_model)
