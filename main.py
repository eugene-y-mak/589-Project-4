import pandas as pd
import NN
import helpers
import numpy as np
import stratified_validation as sv

# wine -- 0
# house -- 1
# cancer -- 2
DATA = 0

if DATA == 0:
    # ------------- For Wine Dataset --------------------- (for later, make these arguments)
    CSV = 'datasets/hw3_wine.csv'
    NAME = "Wine"
    LABEL_HEADER = '# class'
    MASTER_DATASET = pd.read_csv(CSV, sep='\t')  # Note: separating character can be different!
    MASTER_DATASET.columns = MASTER_DATASET.columns.map(str)
    CATEGORICALS = []
    K = 10
    # 1 layer:
    # 8 is good, >8 is HORRIBLE??
    HIDDEN_LAYER_STRUCTURE = [8]
    ALPHA = 1
    EPSILON = 10e-7
    REG_LAMBDA = 0

elif DATA == 1:
    CSV = 'datasets/hw3_house_votes_84.csv'
    NAME = "House Votes"
    LABEL_HEADER = 'class'
    MASTER_DATASET = pd.read_csv(CSV)
    CATEGORICALS = []
    K = 10
    HIDDEN_LAYER_STRUCTURE = [16]
    ALPHA = 2
    EPSILON = -float('inf')  # 10e-8 -float('inf')
    REG_LAMBDA = 0


def main():
    column_names = MASTER_DATASET.columns.to_numpy().copy()
    all_attributes = np.delete(column_names, np.where(column_names == LABEL_HEADER))
    normalized_df = helpers.normalize_dataset(MASTER_DATASET)
    possible_class_labels = helpers.get_attribute_values(MASTER_DATASET, LABEL_HEADER)
    # create folds
    folds = sv.create_k_folds(K, normalized_df, LABEL_HEADER)

    # after making folds, process each one to have one hot encoding class labels for training
    # also, normalize the data
    for i in range(len(folds)):
        folds[i] = helpers.encode_attribute(folds[i], LABEL_HEADER)
    assert K == len(folds)
    # TODO: for CMC EC, one hot encode for categoricals with string elements

    print(f"Final accuracy: {sv.evaluate_NN(LABEL_HEADER, K, folds, HIDDEN_LAYER_STRUCTURE, ALPHA, EPSILON, REG_LAMBDA)}")
    # TODO:
    #  2.) When checking with label, need to figure out how to check for TP vs FP
    return 0


if __name__ == "__main__":
    main()
