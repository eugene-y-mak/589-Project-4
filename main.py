import os

import pandas as pd
import NN
import helpers
import numpy as np
import stratified_validation as sv
from sklearn.model_selection import train_test_split

# wine -- 0
# house -- 1
# cancer -- 2
# CMC -- 3
DATA = 1

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
    CSV = 'datasets/hw3_wine.csv'
    # CSV = "/Users/eugenemak/PycharmProjects/589-Project-4/datasets/hw3_wine.csv"
    NAME = "Wine"
    LABEL_HEADER = '# class'
    MASTER_DATASET = pd.read_csv(CSV, sep='\t')  # Note: separating character can be different!
    MASTER_DATASET.columns = MASTER_DATASET.columns.map(str)
    CATEGORICALS = []
    K = 10
    HIDDEN_LAYER_STRUCTURE = [16, 8, 4, 2]
    ALPHA = 1e-1
    EPSILON = 10e-8
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
    HIDDEN_LAYER_STRUCTURE = [8]
    ALPHA = 1
    EPSILON = 10e-7
    REG_LAMBDA = 0


def main():
    # normalize data
    normalized_df = helpers.normalize_dataset(MASTER_DATASET)
    possible_class_labels = helpers.get_attribute_values(normalized_df, LABEL_HEADER)

    # one hot encode entire dataset
    normalized_OHE_df = helpers.encode_attribute(normalized_df, LABEL_HEADER)

    # OHE categoricals
    for category in CATEGORICALS:
        normalized_OHE_df = helpers.encode_attribute(normalized_OHE_df, category)

    # create folds from one hot version
    folds = sv.create_k_folds(K, normalized_OHE_df, LABEL_HEADER, possible_class_labels)

    assert K == len(folds)
    accuracy, F1 = sv.evaluate_NN(LABEL_HEADER, K, folds, HIDDEN_LAYER_STRUCTURE, ALPHA, EPSILON, REG_LAMBDA)
    print(f"Final accuracy: {accuracy}, F1: {F1}")

    # creating learning curve graph
    train_set, test_set = train_test_split(normalized_OHE_df, test_size=0.2)
    return 0


if __name__ == "__main__":
    main()
