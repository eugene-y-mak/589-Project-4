import pandas as pd
import NN
import helpers
import numpy as np
import stratified_validation as sv

# ------------- For Wine Dataset --------------------- (for later, make these arguments)
CSV = 'datasets/hw3_wine.csv'
NAME = "Wine"
LABEL_HEADER = '# class'
MASTER_DATASET = pd.read_csv(CSV, sep='\t')  # Note: separating character can be different!

# CSV = 'datasets/hw3_house_votes_84.csv'
# NAME = "House Votes"
# LABEL_HEADER = 'class'
# MASTER_DATASET = pd.read_csv(CSV)
MASTER_DATASET.columns = MASTER_DATASET.columns.map(str)
CATEGORICALS = []
K = 10
HIDDEN_LAYER_STRUCTURE = [7, 7]
#ALPHA = 1 / (10 ** 2)
ALPHA = 1
EPSILON = 10e-100
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
    # input layer length must be equal to number of attributes
    # output layer length must be equal to number of classes
    num_layers = len(HIDDEN_LAYER_STRUCTURE) + 2  # add 2 more for input and output
    thetas = NN.make_random_weights(HIDDEN_LAYER_STRUCTURE, len(all_attributes), len(possible_class_labels))

    # TODO: fix this later obv, don't just use 1 fold iteration
    test_set = folds[0]
    train_set = []
    for i in range(1, len(folds)):
        train_set.append(folds[i])
    train_set = pd.concat(train_set)
    input_labels = [col for col in train_set.columns if LABEL_HEADER not in col]
    output_labels = [col for col in train_set.columns if LABEL_HEADER in col]
    true_thetas = NN.train_NN(alpha=ALPHA, epsilon=EPSILON, reg_lambda=REG_LAMBDA,
                              num_layers=num_layers, thetas=thetas, trainings=train_set,
                              input_label=input_labels, output_label=output_labels)

    predictions = test_set.apply(sv.predict_with_NN, args=(input_labels, num_layers, true_thetas,), axis=1)
    actual = pd.Series((test_set[output_labels]).values.tolist())
    assert len(actual) != 0
    assert len(predictions) == len(actual)
    # TODO:
    #  1.) setup prediction function. Using best weights,
    #  run forward prop for every training instance using those weights.
    #  For each run, check prediction, apply argmax, check with label.
    #  2.) When checking with label, need to figure out how to check for TP vs FP
    return 0


if __name__ == "__main__":
    main()
