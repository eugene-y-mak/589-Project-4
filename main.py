import pandas
import helpers
import numpy as np
import stratified_validation as sv


CSV = 'datasets/hw3_house_votes_84.csv'
NAME = "House Votes"
LABEL_HEADER = 'class'
MASTER_DATASET = pandas.read_csv(CSV)
MASTER_DATASET.columns = MASTER_DATASET.columns.map(str)
POSSIBLE_CLASS_LABELS = helpers.get_attribute_values(MASTER_DATASET, LABEL_HEADER)
K = 10


def main():
    # column_names = MASTER_DATASET.columns.to_numpy().copy()
    # all_attributes = np.delete(column_names, np.where(column_names == LABEL_HEADER))
    folds = sv.create_k_folds(K, MASTER_DATASET, POSSIBLE_CLASS_LABELS, LABEL_HEADER)
    assert K == len(folds)
    return 0


if __name__ == "__main__":
    main()
