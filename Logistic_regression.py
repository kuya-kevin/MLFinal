#TODO: check for redundant packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# deal with imbalanced data
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# perform PCA on X, finding PCs that explains "percent" of data
# Output: X_reduced, X with reduced dimension
def PCA_X(X, percent):
    feature_cols = [x for x in range(5408)]     # store the features by their indexes
    feature_cols_np = np.array(feature_cols)

    mean_center_X = X - np.mean(X, axis = 0)    # get the mean centered X from X

    # calculate the covar_matrix
    covar_matrix = mean_center_X.T @ mean_center_X / (len(mean_center_X) - 1)

    # perform eigendecomposition, getting eig_val and eig_vector
    eig_val, eig_vector = eig(covar_matrix)
    print("eig_val", eig_val)
    print("eig_vector", eig_vector)
    print("len eig_val", len(eig_val))
    print("len eig_vector", len(eig_vector))

    # sort through eigen_val, creating "indexes"
    sorted_indexes = eig_val.argsort()[::-1][:len(eig_val)]
    print("sorted_indexes", sorted_indexes)
    eig_val = eig_val[sorted_indexes]

    eig_vector = eig_vector[:,sorted_indexes]   # sort the eig_vector based on sorted_indexes
    feature_cols_np = feature_cols_np[sorted_indexes] # sort the feature_cols based on sorted_indexes

    sum_eig = sum(eig_val)                      # sum over all eig_val for determining percent of variability

    # up toward what number of principle components does 95% of data's variability get explained
    count = 0
    sum_eig_sofar = 0
    for i in range(len(eig_val)):
        if sum_eig_sofar < (percent * sum_eig):
            sum_eig_sofar += eig_val[i]
            count += 1

    print("count", count)
    print("feature_cols_np", feature_cols_np)
    #for i in range(count):
    #    print(feature_cols_np[i])

    # get eig_vectors that explains "percent" of data
    eig_vector_reduced = eig_vector[:, 0:count]

    # get X_reduced by projecting each data point in X to the M dimensions described by M eigenvectors
    # Note: M here is the amount of eigenvectors that explains "percent" of data
    X_reduced = mean_center_X @ eig_vector_reduced
    return X_reduced


class LogisticRegression:
    def __init__(self, learn_rate = 0.001, num_iters=1000):
        self.learn_rate = learn_rate
        self.num_iters = num_iters
        self.W = None
        self.bias = None

    # X is num_samples by num_features
    # y is 1D row vector for each training sample
    def fit(self, X, y):
        # init params (as zeros)
        num_samples, num_features = X.shape
        self.W = np.zeros(num_features)
        self.bias = 0
        #print("num_samples, num_features", num_samples, num_features)
        #print("self.W.shape", self.W.shape)

        # gradient descent
        for i in range(self.num_iters):
            linear_model = np.dot(X, self.W) + self.bias

            y_predicted = self._sigmoid(linear_model)

            # derivatives
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            # update weights and bias
            self.W -= self.learn_rate * dw
            self.bias -= self.learn_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.W) + self.bias
        y_predicted = self._sigmoid(linear_model)

        # based on y_predicted, get the predicted class label
        y_predicted_label = [1 if i > 0.5 else 0 for i in y_predicted]

        return y_predicted_label

    # sigmoid func
    def _sigmoid(self, x):
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid

def accuracy(y_observed, y_predicted):
    accuracy = np.sum(y_observed == y_predicted) / len(y_observed)
    return accuracy


# first process the imbalanced data
def process_imb_data(X, y, testSize):
    # use train_test_split function to randomly split the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 1234)
    print("X_train.shape", X_train.shape)

    # define pipeline
    over = SMOTE(sampling_strategy = 0.1)
    under = RandomUnderSampler(sampling_strategy = 0.5)
    Pipeline_steps = [('o', over), ('u', under)]
    pipeline = Pipeline(Pipeline_steps)

    X_train_smote, y_train_smote = pipeline.fit_resample(X_train, y_train)
    smote_count_1 = 0
    smote_count_0 = 0

    for i in y_train_smote:
        if i == 1:
            smote_count_1 += 1
        elif i == 0:
            smote_count_0 += 1
    print("smote_count_1, smote_count_0", smote_count_1, smote_count_0)

    return X_train_smote, y_train_smote, X_test, y_test

def LogisticRegression_calc(X, y, testSize):
    X_train_smote, y_train_smote, X_test, y_test = process_imb_data(X, y, testSize)

    Logistic_regressor = LogisticRegression(learn_rate = 0.001, num_iters=1000)
    Logistic_regressor.fit(X, y)

    y_predictions = Logistic_regressor.predict(X_test)
    print("Logistic classification accurary:", accuracy(y_test, y_predictions))
    print("precision_score", precision_score(y_test, y_predictions))
    print("recall_score", recall_score(y_test, y_predictions))
    print("f1_score", f1_score(y_test, y_predictions))
    cm = confusion_matrix(y_test, y_predictions)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
    plt.show()
