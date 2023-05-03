import numpy as np
import math


def sigmoid(val): return 1 / (1 + math.e ** (-val))  # sigmoid function


# computes final output for one training instance
def forward_propagation(num_layers, thetas, training_inst):
    # initialize value of first neuron (a^l=1), first layer
    a = np.array(training_inst["x"])  # x is input, y is output
    assert (a.ndim == 1)  # ensure 'a' is always a vector. This is so that we know we don't need to transpose
    for k in range(0, num_layers - 1):  # k+2 is layer num if layers index from 1
        a = np.insert(a, 0, 1, axis=0)  # insert 1 in front of array, for inserting bias term
        print(f"a: {a}")
        print(f"----------Layer number: {k + 2}----------")
        z = np.matmul(thetas[k], a)
        print(f"z: {z}")
        a = sigmoid(z)
    return a


def cost(reg_lambda, num_layers, thetas, trainings):
    j_sum = 0
    for i in range(len(trainings)):
        training_inst = trainings[i]
        print(f"-----------------------------Training Instance {i+1}-----------------------------")
        output = forward_propagation(num_layers, thetas, training_inst)
        y = np.array(training_inst["y"])
        print(f"Predicted output: {output}")
        print(f"Expected output: {y}")
        assert(y.ndim == 1)  # ensure training instance is just vector, since it's the last layer
        j = (-1 * y) * np.log(output) - ((np.ones(y.size)-y) * np.log(np.ones(y.size)-output))
        assert(j.ndim == 1)  # j should also be just a vector since y and output are vecs
        j = np.sum(j)
        print(f"Cost J: {j}")
        j_sum += j
    n = len(trainings)
    j_sum /= n
    S = 0  # sum of squares of all weights of the network besides bias weights
    for weights in thetas:
        S += np.sum(np.square(np.delete(weights, [0], 1)))
    S *= reg_lambda/(2 * n)
    print(f"Final (regularized) cost, J, based on the complete training set: {j_sum + S}")
    return j_sum + S

