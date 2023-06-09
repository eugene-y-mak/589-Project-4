import sklearn
import pandas as pd
import NN
import helpers


def separate_dataset_by_class(dataset, total_count, possible_class_labels, label_header):
    datasets = []
    ratios = {}
    for label in possible_class_labels:
        new_dataset = sklearn.utils.shuffle(dataset[dataset[f"{label_header}_{label}"] == 1].copy())  # get copy of slice
        datasets.append(new_dataset)
        ratios[label] = len(new_dataset) / total_count
    return datasets, ratios


def create_k_folds(k, dataset, label_header, possible_class_labels):
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


# uses forward prop to make a prediction
def predict_with_NN(row, input_labels, num_layers, true_thetas):
    training_inst = row[input_labels].to_numpy()
    output, _ = NN.forward_propagation(training_inst, num_layers, true_thetas, False)
    return output


def evaluate_NN(label_header, K, folds, hidden_layer_structure, alpha, epsilon, reg_lambda):
    accuracies = 0
    F1s = 0
    for i in range(K):
        print(f"----------------------------Fold {i+1}-----------------------------")
        test_set = folds[i]
        train_set = []
        for j in range(K):
            if j != i:  # concat all datasets except ith
                train_set.append(folds[j])
        train_set = pd.concat(train_set)
        # input layer length must be equal to number of attributes
        # output layer length must be equal to number of classes
        input_labels = [col for col in train_set.columns if label_header not in col]
        output_labels = [col for col in train_set.columns if label_header in col]
        # make new thetas here so that we don't reuse the same thetas everytime
        thetas = NN.make_random_weights(hidden_layer_structure, len(input_labels), len(output_labels))
        num_layers = len(hidden_layer_structure) + 2  # add 2 more for input and output
        # ----train the model-----
        true_thetas = NN.train_NN(alpha=alpha, epsilon=epsilon, reg_lambda=reg_lambda, num_layers=num_layers,
                                  thetas=thetas, trainings=train_set, input_label=input_labels,
                                  output_label=output_labels)

        # ----evaluate model------
        predictions = test_set.apply(predict_with_NN, args=(input_labels, num_layers, true_thetas,), axis=1)
        actual = pd.Series((test_set[output_labels]).values.tolist())
        assert len(actual) != 0
        assert len(predictions) == len(actual)
        # ---------------------------METRICS--------------------------
        num_classes = len(output_labels)

        if num_classes > 2:  # if multiclass
            accuracy = 0
            F1 = 0
            for argmax_index in range(num_classes):
                acc, f1 = helpers.calculate_metrics(actual, predictions, argmax_index)
                accuracy += acc
                F1 += f1
            # averaging metrics for every possible class being the positive one
            accuracy /= num_classes
            F1 /= num_classes
        else:
            accuracy, F1 = helpers.calculate_metrics(actual, predictions, 0)
        accuracies += accuracy
        F1s += F1
        print(f"Accuracy: {accuracy}")
        print(f"F1: {F1}")
    return (accuracies / K), (F1s / K)
