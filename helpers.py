import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


def encode_attribute(dataset, header_name):
    return pd.get_dummies(dataset, columns=[header_name], dtype=int)


def normalize_dataset(df):  # normalizes and does one hot encoding
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df


# gets all possible unique values from given attribute header/name in full dataset.
def get_attribute_values(master_dataset, label_header):
    return np.unique(master_dataset.loc[:, label_header])  # guaranteed to be sorted!


# actuals and predictions are series
# returns accuracy, precision, recall, F1 for predictions with specified positive class
def calculate_metrics(actual, predictions, positive_class_label):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    precision = 0
    recall = 0
    F1 = 0
    for index in actual.index:
        if actual[index] == positive_class_label:
            if actual[index] == predictions[index]:
                TP += 1
            else:
                FN += 1
        else:
            if actual[index] == predictions[index]:
                TN += 1
            else:
                FP += 1
    if TP + FP != 0:
        precision = (TP / (TP + FP))
    if TP + FN != 0:
        recall = (TP / (TP + FN))
    if precision + recall != 0:
        F1 = (2 * ((precision * recall) / (precision + recall)))
    return ((TP + TN) / len(actual)), F1
