# this file needs to be in the same directory as "cleaned_K8.csv"
# assuming we've already run clean_data.py: run with "python3 driver.py"
import pandas as pd
import numpy as np

#import py methods
from logistic_regression import *
from knn import *
from svm import *


#GENERAL QUESTION: can we do data visualizations within each method or within the driver function?

def main():
    df = pd.read_csv("cleaned_K8.csv", header = None, low_memory = False)  # process the data with pandas

    #TODO: check that each method produces same output as ipynb. 

    # #logistic regression start
    feature_cols = [x for x in range(5408)]
    X = df[feature_cols]
    y = df[5408]
    print("dimensions of X", X.shape)
    print(LogisticRegression_calc(X, y))

    X_reduced = PCA_X(X, 0.99)
    print("X_reduced", X_reduced)
    print("X_reduced.shape", X_reduced.shape)
    print("X.shape", X.shape)
    print(LogisticRegression_calc(X_reduced, y))
    
    # #knn start
    acc_knn = KNN2(df,3,0.7)
    print(acc_knn)
    knn_slow = KNN(df, 3, 0.7)
    print(knn_slow)
    # get accurancy when k = 3,5,7,9,...,27
    accs = []
    for i in range(1,15):
        k = 2*i + 1
        print(k)
        acc_knn = KNN2(df,k,0.7)
        accs.append(acc_knn)

    #plotting
    x = [2*i+1 for i in range(1,15)]
    y = accs
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accurancy')
    plt.title('Accurancy as the K goes up')
    plt.plot(x,y)
    plt.show()

    #SVM start
    acc_svm = SVM(df,0.7)
    print(acc_svm)
    

if __name__ == "__main__":
    main()
