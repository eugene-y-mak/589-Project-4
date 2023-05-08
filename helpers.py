import numpy as np
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
# returns accuracy, F1 for predictions with specified positive class
def calculate_metrics(actual, predictions, positive_class_label):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    precision = 0
    recall = 0
    F1 = 0
    for (a, p) in zip(actual, predictions):
        predicted_class_index = np.argmax(p)
        prediction_array = np.zeros(3)
        prediction_array[predicted_class_index] = 1
        actual_class_index = np.argmax(a)
        if actual_class_index == positive_class_label:
            if predicted_class_index == actual_class_index:
                TP += 1
            else:
                FN += 1
        else:
            if predicted_class_index == actual_class_index:
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
