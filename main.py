import pandas

import NN
import helpers
import numpy as np
import stratified_validation as sv


CSV = 'datasets/hw3_house_votes_84.csv'
NAME = "House Votes"
LABEL_HEADER = 'class'
MASTER_DATASET = pandas.read_csv(CSV)
MASTER_DATASET.columns = MASTER_DATASET.columns.map(str)
POSSIBLE_CLASS_LABELS = helpers.get_attribute_values(MASTER_DATASET, LABEL_HEADER)
NUMERICALS = []
K = 10
HIDDEN_LAYER_STRUCTURE = [10, 15, 10]


def main():
    column_names = MASTER_DATASET.columns.to_numpy().copy()
    all_attributes = np.delete(column_names, np.where(column_names == LABEL_HEADER))
    normalized_df = helpers.preprocess_for_NN(MASTER_DATASET, NUMERICALS)
    # preprocess dataset to convert categoricals to one hot encoding
    # create folds
    folds = sv.create_k_folds(K, normalized_df, POSSIBLE_CLASS_LABELS, LABEL_HEADER)
    assert K == len(folds)

    # input layer length must be equal to number of attributes
    # output layer length must be equal to number of classes
    num_layers = len(HIDDEN_LAYER_STRUCTURE) + 2  # add 2 more for input and output
    thetas = NN.make_random_weights(HIDDEN_LAYER_STRUCTURE, len(all_attributes), len(POSSIBLE_CLASS_LABELS))
    print(thetas)
    return 0


if __name__ == "__main__":
    main()
