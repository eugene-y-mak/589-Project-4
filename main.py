import os

import pandas as pd
import NN
import helpers
import numpy as np
import stratified_validation as sv

# wine -- 0
# house -- 1
# cancer -- 2
DATA = 0

# structures to test:
# [4] lambda=0
# [8] lambda=0
# [16] lambda=0
# [32] lambda=0
# [8, 4] lambda=0
# [4, 8] lambda=0
# [32, 16] lambda=0
# [16, 32] lambda=0
# [8, 4, 2] lambda=0
# [2, 4, 8] lambda=0
# [16, 8, 4, 2] lambda=0

# [8] lambda = 0.25
# [8, 4] lambda = 0.25
# [8] lambda = 0.5? or 1
# [8, 4] lambda = 0.5 or 1

if DATA == 0:
    # ------------- For Wine Dataset --------------------- (for later, make these arguments)
    #CSV = 'datasets/hw3_wine.csv'
    CSV = "/Users/eugenemak/PycharmProjects/589-Project-4/datasets/hw3_wine.csv"
    NAME = "Wine"
    LABEL_HEADER = '# class'
    MASTER_DATASET = pd.read_csv(CSV, sep='\t')  # Note: separating character can be different!
    MASTER_DATASET.columns = MASTER_DATASET.columns.map(str)
    CATEGORICALS = []
    K = 10
    HIDDEN_LAYER_STRUCTURE = [9]
    ALPHA = 1
    EPSILON = 10e-7
    REG_LAMBDA = 0

elif DATA == 1:
    #CSV = 'datasets/hw3_house_votes_84.csv'
    CSV = "/Users/eugenemak/PycharmProjects/589-Project-4/datasets/hw3_house_votes_84.csv"
    NAME = "House Votes"
    LABEL_HEADER = 'class'
    MASTER_DATASET = pd.read_csv(CSV)
    CATEGORICALS = []
    K = 10
    HIDDEN_LAYER_STRUCTURE = [16]
    ALPHA = 2
    EPSILON = 10e-8  # 10e-8 -float('inf')
    EPOCHS = 500  # unused, for now
    REG_LAMBDA = 0

elif DATA == 2:
    CSV = "/Users/eugenemak/PycharmProjects/589-Project-4/datasets/hw3_cancer.csv"
    NAME = "Cancer"
    LABEL_HEADER = 'Class'
    MASTER_DATASET = pd.read_csv(CSV, sep='\t')  # Note: separating character can be different!
    MASTER_DATASET.columns = MASTER_DATASET.columns.map(str)
    CATEGORICALS = []
    K = 10
    HIDDEN_LAYER_STRUCTURE = [8]
    ALPHA = 1
    EPSILON = 10e-7
    REG_LAMBDA = 0

def main():
    # normalize data
    normalized_df = helpers.normalize_dataset(MASTER_DATASET)

    # create folds
    folds = sv.create_k_folds(K, normalized_df, LABEL_HEADER)

    # TODO: for CMC EC, one hot encode for categoricals with string elements
    # after making folds, process each one to have one hot encoding class labels for training
    for i in range(len(folds)):
        folds[i] = helpers.encode_attribute(folds[i], LABEL_HEADER)
    assert K == len(folds)
    accuracy, F1 = sv.evaluate_NN(LABEL_HEADER, K, folds, HIDDEN_LAYER_STRUCTURE, ALPHA, EPSILON, REG_LAMBDA)
    print(f"Final accuracy: {accuracy}, F1: {F1}")
    return 0


if __name__ == "__main__":
    main()
