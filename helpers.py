import numpy as np
import pandas as pd
from sklearn import preprocessing

def encode_attribute(dataset, header_name):
    return pd.get_dummies(dataset, columns=[header_name], dtype=int)


def normalize_dataset(df):  # normalizes
    # print(df.min())
    # print(df.max())
    # normalized_df = 0 if (df.max() - df.min()) == 0 else (df - df.min()) / (df.max() - df.min())
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df


# gets all possible unique values from given attribute header/name in full dataset.
def get_attribute_values(master_dataset, label_header):
    return np.unique(master_dataset.loc[:, label_header])  # guaranteed to be sorted!


# actuals and predictions are series
# returns accuracy, F1 for predictions with specified positive class
def calculate_metrics(actual, predictions, positive_class_label):
    print(positive_class_label)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    precision = 0
    recall = 0
    F1 = 0
    for (a, p) in zip(actual, predictions):
        predicted_class_index = np.argmax(p)
        # print(f"Predicted: {predicted_class_index}")
        prediction_array = np.zeros(len(p))
        prediction_array[predicted_class_index] = 1
        actual_class_index = np.argmax(a)
        # print(f"Actual: {actual_class_index}")
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
    accuracy = ((TP + TN) / len(actual))
    print(f"ACCURACY AT END OF CALCULATE_METRICS: {accuracy} ")
    return accuracy, F1
