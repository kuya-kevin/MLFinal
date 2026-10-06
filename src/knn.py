#TODO: check for redundant packages

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



# splitData is a function that takes a dataframe df, and return the X dataset and y dataset
def SplitData(df): 
    X = df.iloc[:,:5408]
    y = df.iloc[:,5408:5409]
    return X, y


def EuclideanDistance(s1,s2):
    x1,y1 = SplitData(s1)
    x2,y2 = SplitData(s2)
    dist = np.sqrt(np.sum(np.square(x1.values - x2.values)))
    return dist


def GetNeighbors(df_train, df_test, k_num):
    distances = []
    for i in range(len(df)):
        train = df.iloc[i:i+1,:]
        dist = EuclideanDistance(train, df_test)
        distances.append((train, dist))
    distances.sort(key=lambda tup: tup[1]) # sort based on dist
    neighbors = [] 
    for i in range(k_num):
        neighbors.append(distances[i][0])
    return neighbors


def FindMajorityLabel(List):
    return max(set(List), key = List.count)


def MakePrediction(df_train, test, k_num):
    neighbors = GetNeighbors(df_train, test, k_num)       #get k_num neighbors of test dataset
    neighbor_values = [n.iloc[0,5408] for n in neighbors] #get label of each neighbor
    prediction = FindMajorityLabel(neighbor_values)       #make prediction based on majority label in neighbors
    return prediction


def GetLabels(df, k_num, train_size):
    labels = []                            # save predict label and true label in a matrix
    len_train = int(train_size*len(df))
    df_train = df.iloc[:len_train,:]
    df_test = df.iloc[len_train:,:]
    for i in range(len(df_test)):
        test = df_test.iloc[i:i+1,:]
        y_pred = MakePrediction(df_train,test,k_num)
        y_true = df_test.iloc[i,5408]
        labels.append((y_pred,y_true))
    return labels


def ComputeAcc(labels):
    total = len(labels)
    correct = 0
    for y in labels:
        if y[0] == y[1]:
            correct += 1
    acc = correct/total
    return acc


def KNN(df, k_num,train_size):
    labels = GetLabels(df,k_num,train_size)
    acc_knn = ComputeAcc(labels)
    return acc_knn


# I use knn package to run different k as it takes less time.
def KNN2(df, k_num,train_size):
    len_train = int(train_size*len(df))
    df_train = df.iloc[:len_train,:]
    df_test = df.iloc[len_train:,:]
    X_train, y_train = SplitData(df_train)
    X_test, y_test = SplitData(df_test)
    neigh = KNeighborsClassifier(n_neighbors=k_num)
    neigh.fit(X_train, np.ravel(y_train))
    y_pred = neigh.predict(X_test)
    y_true = y_test
    acc_knn = accuracy_score(y_pred, y_true)
    return acc_knn

# SVM

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

# %%
df = pd.read_csv("cleaned_K8.csv", header = None)

# %%
acc_svm = SVM(df,0.7)

# %%
acc_svm




