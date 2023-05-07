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
    normalized_df = helpers.normalize_dataset(MASTER_DATASET)

    # normalized_encoded_df = helpers.encode_attribute(normalized_df, LABEL_HEADER)
    # create folds
    folds = sv.create_k_folds(K, normalized_df, POSSIBLE_CLASS_LABELS, LABEL_HEADER)

    # after making folds, process each one to have one hot encoding class labels for training later
    for i in range(len(folds)):
        folds[i] = helpers.encode_attribute(folds[i], LABEL_HEADER)
    assert K == len(folds)

    # input layer length must be equal to number of attributes
    # output layer length must be equal to number of classes
    num_layers = len(HIDDEN_LAYER_STRUCTURE) + 2  # add 2 more for input and output
    thetas = NN.make_random_weights(HIDDEN_LAYER_STRUCTURE, len(all_attributes), len(POSSIBLE_CLASS_LABELS))

    #  1.) one hot encode the labels... using scikit or numpy get_dummies
    # TODO:
    #  2.) change NN processing to take in a dataframe, not hashmap. Label header
    #  becomes "y", everything else in that row becomes the input "x".
    #  Question: if label column becomes multiple from ohe, how to handle?
    #  3.) iterate through all training instances, being rows of the dataframe
    #  4.)  Test using examples by changing input to be a dataframe
    #  5.) Once that works, test on house votes and check if cost is going down and is training
    #  6.) After that, setup prediction function using argmax of output of NN, checking with label
    return 0


if __name__ == "__main__":
    main()
