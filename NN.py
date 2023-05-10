import numpy as np
import math
import pandas as pd


def sigmoid(val): return 1 / (1 + math.e ** (-val))  # sigmoid function


# computes final output for one training instance
def forward_propagation(training_inst, num_layers, thetas, do_print):
    # initialize value of first neuron (a^l=1), first layer
    a = training_inst  # x is input, y is output
    assert (a.ndim == 1)  # ensure 'a' is always a vector. This is so that we know we don't need to transpose
    # assert (num_layers == len(thetas) + 1)  # make sure num_layers is correct
    activations = []
    for k in range(0, num_layers - 1):  # k+2 is layer num if layers index from 1
        a = np.insert(a, 0, 1, axis=0)  # insert 1 in front of array, for inserting bias term
        activations.append(np.array([a.copy()]))  # force to be a row vector
        if do_print:
            print(f"a: {a}")
            print(f"----------Layer number: {k + 2}----------")
        z = np.matmul(thetas[k], a)
        if do_print: print(f"z: {z}")
        a = sigmoid(z)
    # don't need bias for last one because not calculating gradient from last layer
    activations.append(np.array([a.copy()]))
    # print(a)
    return a, activations


def cost(reg_lambda, num_layers, thetas, trainings, input_label, output_label, do_print):
    if do_print: print("-----------------------------------Computing error/cost J of the "
                       "network----------------------------------------")
    j_sum = 0
    i = 0
    # Convert dataframe columns of input or output label(s) to a series of df rows in form of lists
    attribute_data = pd.Series((trainings[input_label]).values.tolist())
    label_data = pd.Series((trainings[output_label]).values.tolist())
    for training_inst in zip(attribute_data, label_data):
        if do_print: print(f"-----------------------------Training Instance {i + 1}-----------------------------")
        output, _ = forward_propagation(np.array(training_inst[0]), num_layers, thetas, do_print)
        y = np.array(training_inst[1])
        if do_print:
            print(f"Predicted output: {output}")
            print(f"Expected output: {y}")
        assert (y.ndim == 1)  # ensure training instance is just vector, since it's the last layer
        j = (-1 * y) * np.log(output) - ((np.ones(y.size) - y) * np.log(np.ones(y.size) - output))
        assert (j.ndim == 1)  # j should also be just a vector since y and output are vecs
        j = np.sum(j)
        if do_print: print(f"Cost J: {j}")
        j_sum += j
        i += 1
    n = len(trainings)
    j_sum /= n
    S = 0  # sum of squares of all weights of the network besides bias weights
    for weights in thetas:
        S += np.sum(np.square(np.delete(weights, [0], 1)))
    S *= reg_lambda / (2 * n)
    if do_print: print(f"Final (regularized) cost, J, based on the complete training set: {j_sum + S}")
    return j_sum + S


# alpha = 1/10^3
# epsilon = 10e-3
# thetas: python list of numpy arrays,corresponding to each theta array
def back_propagation(alpha, reg_lambda, num_layers, thetas, trainings, input_label, output_label, do_print):
    if do_print: print("----------------------------------------------Running back "
                       "propagation----------------------------------------------")
    # print(f"Diff: {diff}")
    accumulated_gradients = {}
    for i in range(len(thetas)):
        accumulated_gradients[i] = None
    inst_num = 1
    attribute_data = pd.Series((trainings[input_label]).values.tolist())
    label_data = pd.Series((trainings[output_label]).values.tolist())
    for training_inst in zip(attribute_data, label_data):
        if do_print: print(f"-----------------------Training Instance {inst_num}-----------------------")
        output, activations = forward_propagation(np.array(training_inst[0]), num_layers, thetas, False)
        y = np.array(training_inst[1])
        if do_print: print("--------------Computing Deltas--------------")
        delta = output - y
        if do_print: print(f"delta{len(thetas) + 1}: {delta}")
        all_deltas = [np.array([delta.copy()]).T]  # ensure column vector!!
        # since len thetas is 1 less than num layers, guaranteed to be for all layers L-1...2 (if start from 1)
        for k in range(len(thetas) - 1, 0, -1):
            # remove first column of thetas, being the bias deltas.
            delta = np.delete(np.matmul(thetas[k].T, delta) * activations[k]
                              * (np.ones(activations[k].size) - activations[k]), 0)
            all_deltas.append(np.array([delta.copy()]).T)  # specify as row vector, then transpose to get col
            if do_print: print(f"delta{k + 1}: {delta}")
        # need to reverse because we added deltas backwards before so 0th index = deltas for last layer
        all_deltas.reverse()
        # activations will have 1 more than deltas, b/c use final layer activation neurons
        assert (len(all_deltas) == len(activations) - 1)
        if do_print: print("---------------Computing Gradients---------------")
        for k in range(num_layers - 2, -1, -1):  # num_layers - 2 means start from second to last layer (L-1)
            new_gradient = np.matmul(all_deltas[k], activations[k])
            if do_print: print(f"theta{k + 1}:\n {new_gradient}")
            if accumulated_gradients[k] is None:
                accumulated_gradients[k] = new_gradient
            else:
                accumulated_gradients[k] += new_gradient
        inst_num += 1
    if do_print: print("-----------Training set finished processing. Compute average (regularized "
                       "gradients)-----------")
    n = len(trainings)
    for k in range(num_layers - 2, -1, -1):
        # computing regularizer
        regularizer = reg_lambda * thetas[k]  # regularizer is P in pseudocode
        regularizer[:, 0] = 0
        accumulated_gradients[k] = (1 / n) * (accumulated_gradients[k] + regularizer)

    if do_print:
        for i in range(len(thetas)):
            print(f"theta{i + 1}:\n {accumulated_gradients[i]}")

    # -----updating weights------
    for k in range(num_layers - 2, -1, -1):
        thetas[k] -= alpha * accumulated_gradients[k]

    new_cost = cost(reg_lambda, num_layers, thetas, trainings, input_label, output_label, False)
    return new_cost, thetas


def numerical_gradients_check(Thetas, eps, Num_layers, Trainings, Input_label, Output_label, Do_print):
    def J(num_layers, thetas, trainings, input_label, output_label, do_print):
        j_sum = 0
        i = 0
        # Convert dataframe columns of input or output label(s) to a series of df rows in form of lists
        attribute_data = pd.Series((trainings[input_label]).values.tolist())
        label_data = pd.Series((trainings[output_label]).values.tolist())
        for training_inst in zip(attribute_data, label_data):
            if do_print: print(f"-----------------------------Training Instance {i + 1}-----------------------------")
            output, _ = forward_propagation(np.array(training_inst[0]), num_layers, thetas, do_print)
            y = np.array(training_inst[1])
            if do_print:
                print(f"Predicted output: {output}")
                print(f"Expected output: {y}")
            assert (y.ndim == 1)  # ensure training instance is just vector, since it's the last layer
            j = (-1 * y) * np.log(output) - ((np.ones(y.size) - y) * np.log(np.ones(y.size) - output))
            assert (j.ndim == 1)  # j should also be just a vector since y and output are vecs
            j = np.sum(j)
            if do_print: print(f"Cost J: {j}")
            j_sum += j
            i += 1
        n = len(trainings)
        j_sum /= n
        return j_sum

    theta_num = 1
    for k in range(len(Thetas)):
        # theta is all weights for layer
        theta = Thetas[k]
        verifier_theta = np.zeros(theta.shape)
        print(f"theta {theta_num}")
        for weight in theta.flat:
            new_thetas = Thetas.copy()
            new_theta = theta.copy()
            i, j = np.where(theta == weight)
            new_theta[i, j] = weight + eps
            new_thetas[k] = new_theta
            first_val = J(Num_layers, new_thetas, Trainings, Input_label, Output_label, Do_print)
            new_theta[i, j] = weight - eps
            new_thetas[k] = new_theta
            second_val = J(Num_layers, new_thetas, Trainings, Input_label, Output_label, Do_print)
            verifier_theta[i, j] = (first_val - second_val) / eps
        theta_num += 1
        print(verifier_theta)


def train_NN(alpha, epsilon, reg_lambda, num_layers, thetas, trainings, input_label, output_label, max_iterations=500):
    assert (num_layers == len(thetas) + 1)  # make sure num_layers is correct
    # stopping criteria:
    # cost function improves by less than epsilon e
    J = cost(reg_lambda, num_layers, thetas, trainings, input_label, output_label, False)
    print(f"Initial cost: {J}")
    diff = float('inf')
    iterations = 0
    while diff > epsilon and iterations < max_iterations:
        # print(f"Cost: {J}")
        new_cost, new_thetas = back_propagation(alpha, reg_lambda, num_layers, thetas,
                                                trainings, input_label, output_label, False)
        diff = abs(J - new_cost)
        J = new_cost
        thetas = new_thetas
        iterations += 1
    print(f"Final cost: {J}")
    return thetas


def train_NN_plot(alpha, epsilon, reg_lambda, num_layers, thetas,
                  trainings, input_label, output_label, test_set, max_iterations=500):
    assert (num_layers == len(thetas) + 1)  # make sure num_layers is correct
    # stopping criteria:
    # cost function improves by less than epsilon e
    J = cost(reg_lambda, num_layers, thetas, trainings, input_label, output_label, False)
    print(f"Initial cost: {J}")
    diff = float('inf')
    iterations = 0
    performance = [J]
    num_trainings = [iterations]
    while diff > epsilon and iterations < max_iterations:
        # print(f"Cost: {J}")
        new_cost, new_thetas = back_propagation(alpha, reg_lambda, num_layers, thetas,
                                                trainings, input_label, output_label, False)
        diff = abs(J - new_cost)
        J = new_cost
        thetas = new_thetas
        iterations += 1
        print(iterations)
        num_trainings.append(iterations * len(trainings.index))
        print(iterations * len(trainings.index))
        # get performance from test_set
        performance.append(cost(reg_lambda, num_layers, thetas, test_set, input_label, output_label, False)
                           )
    print(f"Final cost: {J}")
    return performance, num_trainings


def make_random_weights(hidden_layer_structure, length_of_input, length_of_output):
    # For thetas:
    # num of rows = num of neurons in next layer
    # num of cols = number of neurons for current layer + bias term
    # first col is bias terms, then weights
    max_val, min_val = 1, -1
    range_size = (max_val - min_val)  # 2

    thetas = []
    # initial step with input layer:
    assert (len(hidden_layer_structure) != 0)
    thetas.append(np.random.rand(hidden_layer_structure[0], length_of_input + 1) * range_size + min_val)
    i = 0
    while i < len(hidden_layer_structure) - 1:  # checking if i is the last hidden layer or not
        thetas.append(
            np.random.rand(hidden_layer_structure[i + 1], hidden_layer_structure[i] + 1) * range_size + min_val)
        i += 1

    # final step with output layer
    thetas.append(np.random.rand(length_of_output, hidden_layer_structure[i] + 1) * range_size + min_val)
    # should have same number of weights as all the layers (including input and output) minus 1,
    # since don't need weights for output layer
    assert (len(thetas) == len(hidden_layer_structure) + 1)
    return thetas
