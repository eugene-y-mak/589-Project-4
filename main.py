import pandas as pd
import NN
import helpers
import numpy as np
import stratified_validation as sv

CSV = 'datasets/hw3_house_votes_84.csv'
NAME = "House Votes"
LABEL_HEADER = 'class'
MASTER_DATASET = pd.read_csv(CSV)
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

    # TODO: fix this later obv, don't just use 1 fold iteration
    test_set = folds[0]
    train_set = []
    for i in range(1, len(folds)):
        train_set.append(folds[i])
    train_set = pd.concat(train_set)
    input_labels = [col for col in train_set.columns if LABEL_HEADER not in col]
    output_labels = [col for col in train_set.columns if LABEL_HEADER in col]
    NN.train_NN(alpha=1 / (10 ** 3), epsilon=10e-4, reg_lambda=0.25, num_layers=len(HIDDEN_LAYER_STRUCTURE)+2,
                thetas=thetas, trainings=train_set,
                input_label=input_labels,
                output_label=output_labels)

    # TODO:
    #  6.) After that, setup prediction function using argmax of output of NN, checking with label
    return 0


if __name__ == "__main__":
    main()
