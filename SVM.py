# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:43:26 2019

@author: mmrra
"""

####SVM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("F:\MachineLearning\Datasets\Titanic/trainpp.csv")
train.head()

target=train["Survived"]
train_data=train.drop(["Survived"], axis=1)
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing, model_selection, metrics
import warnings
warnings.filterwarnings('ignore')
k_fold = KFold(n_splits = 10, shuffle=True, random_state=0)

support = svm.LinearSVC(random_state=20)
scoring='accuracy'

score = cross_val_score(support, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
print(round(np.mean(score)*100, 2))

conf_mat = metrics.confusion_matrix(train_data, target)