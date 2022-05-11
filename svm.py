#TODO: check for redundant packages

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# %%
def SVM(df, split_ratio):
    #make sure that data is shuffled to avoid bias
    size_train = int(split_ratio * len(df))
    df_train, df_test = df.head(size_train), df.iloc[size_train:,:]
    
    X_train, y_train = df_train.iloc[:,:df_train.shape[1] - 1], df_train.iloc[:,df_train.shape[1] - 1]
    X_test, y_test = df_test.iloc[:,:df_test.shape[1] - 1], df_test.iloc[:,df_test.shape[1] - 1]
    print(X_train, y_train)
    
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    return accuracy_score(y_test, predicted)

