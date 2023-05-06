import numpy as np
import math


def sigmoid(val): return 1 / (1 + math.e ** (-val))  # sigmoid function


# computes final output for one training instance
def forward_propagation(num_layers, thetas, training_inst, do_print):
    # initialize value of first neuron (a^l=1), first layer
    a = np.array(training_inst["x"])  # x is input, y is output
    assert (a.ndim == 1)  # ensure 'a' is always a vector. This is so that we know we don't need to transpose
    assert (num_layers == len(thetas) + 1)  # make sure num_layers is correct
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
    return a, activations


def cost(reg_lambda, num_layers, thetas, trainings, do_print):
    if do_print: print("-----------------------------------Computing error/cost J of the "
                       "network----------------------------------------")
    j_sum = 0
    for i in range(len(trainings)):
        training_inst = trainings[i]
        if do_print: print(f"-----------------------------Training Instance {i + 1}-----------------------------")
        output, _ = forward_propagation(num_layers, thetas, training_inst, do_print)
        y = np.array(training_inst["y"])
        if do_print:
            print(f"Predicted output: {output}")
            print(f"Expected output: {y}")
        assert (y.ndim == 1)  # ensure training instance is just vector, since it's the last layer
        j = (-1 * y) * np.log(output) - ((np.ones(y.size) - y) * np.log(np.ones(y.size) - output))
        assert (j.ndim == 1)  # j should also be just a vector since y and output are vecs
        j = np.sum(j)
        if do_print: print(f"Cost J: {j}")
        j_sum += j
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
def back_propagation(alpha, epsilon, max_iterations, reg_lambda, num_layers, thetas, trainings, do_print):
    if do_print: print("----------------------------------------------Running back "
                       "propagation----------------------------------------------")
    assert (num_layers == len(thetas) + 1)  # make sure num_layers is correct
    # stopping criteria:
    # cost function improves by less than epsilon e
    J = cost(reg_lambda, num_layers, thetas, trainings, False)
    diff = float('inf')
    iterations = 0
    while diff > epsilon and iterations < max_iterations:
        print(f"Cost: {J}")
        # print(f"Diff: {diff}")
        accumulated_gradients = {}
        for i in range(len(thetas)):
            accumulated_gradients[i] = None
        for i in range(len(trainings)):
            training_inst = trainings[i]
            if do_print: print(f"-----------------------Training Instance {i + 1}-----------------------")
            output, activations = forward_propagation(num_layers, thetas, training_inst, False)
            y = np.array(training_inst["y"])
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
        diff = J - cost(reg_lambda, num_layers, thetas, trainings, False)
        J = cost(reg_lambda, num_layers, thetas, trainings, False)
        iterations += 1
    return 0
