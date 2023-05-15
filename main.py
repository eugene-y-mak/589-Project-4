import os

import pandas as pd
import NN
import helpers
import numpy as np
import stratified_validation as sv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
# wine -- 0
# house -- 1
# cancer -- 2
# CMC -- 3
DATA = 4

if DATA == 0:
    # ------------- For Wine Dataset --------------------- (for later, make these arguments)
    CSV = 'datasets/hw3_wine.csv'
    # CSV = "/Users/eugenemak/PycharmProjects/589-Project-4/datasets/hw3_wine.csv"
    NAME = "Wine"
    LABEL_HEADER = '# class'
    MASTER_DATASET = pd.read_csv(CSV, sep='\t')  # Note: separating character can be different!
    MASTER_DATASET.columns = MASTER_DATASET.columns.map(str)
    CATEGORICALS = []
    K = 10
    HIDDEN_LAYER_STRUCTURE = [32]
    EPOCHS = 500
    ALPHA = 1e-1
    EPSILON = 10e-4
    REG_LAMBDA = 0

elif DATA == 1:
    CSV = 'datasets/hw3_house_votes_84.csv'
    # CSV = "/Users/eugenemak/PycharmProjects/589-Project-4/datasets/hw3_house_votes_84.csv"
    NAME = "House Votes"
    LABEL_HEADER = 'class'
    MASTER_DATASET = pd.read_csv(CSV)
    # CATEGORICALS = []
    COLUMN_NAMES = MASTER_DATASET.columns.to_numpy().copy()
    CATEGORICALS = np.delete(COLUMN_NAMES, np.where(COLUMN_NAMES == LABEL_HEADER))
    K = 10
    HIDDEN_LAYER_STRUCTURE = [16]
    ALPHA = 10e-1
    EPSILON = 10e-4  # 10e-8 -float('inf')
    EPOCHS = 500
    REG_LAMBDA = 0

elif DATA == 2:
    CSV = 'datasets/hw3_cancer.csv'
    # CSV = "/Users/eugenemak/PycharmProjects/589-Project-4/datasets/hw3_cancer.csv"
    NAME = "Cancer"
    LABEL_HEADER = 'Class'
    MASTER_DATASET = pd.read_csv(CSV, sep='\t')  # Note: separating character can be different!
    MASTER_DATASET.columns = MASTER_DATASET.columns.map(str)
    CATEGORICALS = []
    K = 10
    HIDDEN_LAYER_STRUCTURE = [32, 16]
    ALPHA = 0.1
    EPSILON = 10e-4
    EPOCHS = 500
    REG_LAMBDA = 0

elif DATA == 3:
    # ------------- For Contraceptive Dataset ---------------------
    CSV = 'datasets/cmc.data'
    NAME = "Contraceptive"
    COLUMN_NAMES = ["Wife's age", "Wife's education", "Husband's education", "Number of children ever born",
                    "Wife's religion", "Wife's now working?", "Husband's occupation", "Standard-of-living index",
                    "Media exposure", "Contraceptive method used"]
    LABEL_HEADER = "Contraceptive method used"
    CATEGORICALS = ["Wife's education", "Husband's education", "Wife's religion", "Wife's now working?",
                    "Husband's occupation", "Standard-of-living index", "Media exposure"]

    MASTER_DATASET = pd.read_csv(CSV, names=COLUMN_NAMES)
    MASTER_DATASET.columns = MASTER_DATASET.columns.map(str)

    K = 10
    HIDDEN_LAYER_STRUCTURE = [16]
    ALPHA = 1
    EPSILON = 10e-4
    REG_LAMBDA = 0
    EPOCHS = 500

elif DATA == 4:
    digits = datasets.load_digits()
    MASTER_DATASET = pd.DataFrame(digits.images.reshape((len(digits.images), -1)))
    labels = digits.target
    NAME = "Digits"
    LABEL_HEADER = '64'
    CATEGORICALS = []
    MASTER_DATASET[LABEL_HEADER] = labels
    MASTER_DATASET.columns = MASTER_DATASET.columns.map(str)
    K = 10
    HIDDEN_LAYER_STRUCTURE = [4, 4, 4, 4, 4]
    ALPHA = 2
    EPSILON = 10e-4
    REG_LAMBDA = 0
    EPOCHS = 500


def main():
    print(NAME)
    # normalize data
    normalized_df = helpers.normalize_dataset(MASTER_DATASET)
    normalized_df.columns = normalized_df.columns.map(str)
    possible_class_labels = helpers.get_attribute_values(normalized_df, LABEL_HEADER)

    # one hot encode entire dataset
    normalized_OHE_df = helpers.encode_attribute(normalized_df, LABEL_HEADER)

    # OHE categoricals
    for category in CATEGORICALS:
        normalized_OHE_df = helpers.encode_attribute(normalized_OHE_df, category)

    # create folds from one hot version
    folds = sv.create_k_folds(K, normalized_OHE_df, LABEL_HEADER, possible_class_labels)

    # ----------------- EVALUATION --------------------
    assert K == len(folds)
    accuracy, F1 = sv.evaluate_NN(LABEL_HEADER, K, folds, HIDDEN_LAYER_STRUCTURE, ALPHA, EPSILON, REG_LAMBDA)
    print(f"Final accuracy: {accuracy}, F1: {F1}")

    # --------------- Learning Curve ------------------
    test_set = folds[0]
    train_set = []
    for j in range(1, K):
        train_set.append(folds[j])
    train_set = pd.concat(train_set)
    input_labels = [col for col in train_set.columns if LABEL_HEADER not in col]
    output_labels = [col for col in train_set.columns if LABEL_HEADER in col]
    thetas = NN.make_random_weights(HIDDEN_LAYER_STRUCTURE, len(input_labels), len(output_labels))
    y_axis_perf, x_axis_trains = NN.train_NN_plot(ALPHA, EPSILON, REG_LAMBDA, len(HIDDEN_LAYER_STRUCTURE) + 2, thetas,
                                                  train_set, input_labels, output_labels, test_set, EPOCHS)
    plt.plot(x_axis_trains, y_axis_perf)
    plt.title(f"{NAME} Dataset Learning Curve")
    plt.ylabel("Performance (J)")
    plt.xlabel("Number of Training Samples")
    plt.show()
    return 0


if __name__ == "__main__":
    main()
