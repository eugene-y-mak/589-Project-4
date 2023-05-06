import sklearn
import pandas as pd


def separate_dataset_by_class(dataset, total_count, possible_class_labels, label_header):
    datasets = []
    ratios = {}
    for label in possible_class_labels:
        # new_dataset = sklearn.utils.shuffle(dataset[dataset[label_header] == label].copy(),random_state=7)
        # get copy of slice
        new_dataset = sklearn.utils.shuffle(dataset[dataset[label_header] == label].copy())  # get copy of slice
        datasets.append(new_dataset)
        ratios[label] = len(new_dataset) / total_count
    return datasets, ratios


def create_k_folds(k, dataset, possible_class_labels, label_header):
    # 1.) divide dataset into several separated by class
    total_count = len(dataset)
    datasets, ratios = separate_dataset_by_class(dataset, total_count, possible_class_labels, label_header)

    # 2.) For each class dataset, evenly distribute to k folds. Combine folds at the end
    # datasets are shuffled, so just need to split by ratio and distribute the remainders
    folds = {}  # maps index/name (starting from 0) of fold to dataframe
    for fold_index in range(k):
        folds[fold_index] = None
    for class_dataset in datasets:
        instances_per_fold = int(len(class_dataset) / k)  # rounds down to nearest integer
        fold_index = 0  # fold indexer
        #  next time use np.array_split.... ;(
        while len(class_dataset.index) != 0:
            if len(class_dataset.index) < instances_per_fold:  # distribute remainder section to each fold
                first_row = class_dataset.iloc[:1]
                folds[fold_index] = pd.concat([first_row, folds[fold_index]])  # dict indexes with int
                class_dataset = class_dataset.drop(first_row.index)
            else:
                if folds[fold_index] is None:  # not initialized yet, first class
                    folds[fold_index] = class_dataset.iloc[:instances_per_fold]
                else:  # if dataframe already exists
                    folds[fold_index] = pd.concat(
                        [class_dataset.iloc[:instances_per_fold], folds[fold_index]])  # dict indexes with int
                class_dataset = class_dataset.drop(class_dataset.iloc[:instances_per_fold].index)
            fold_index += 1
            fold_index = fold_index % k
    return folds


def evaluate_NN():
    return 0
