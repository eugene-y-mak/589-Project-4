import numpy as np
import math


def sigmoid(val): return 1 / (1 + math.e ** (-val))  # sigmoid function


# computes final output for one training instance
def forward_propagation(num_layers, thetas, training_inst):
    print(f"--------------------Training Instance--------------------")
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
    print(f"final output: {a}")
    return a


def cost(reg_lambda, num_layers, thetas, trainings):
    for training_inst in trainings:
        output = forward_propagation(num_layers, thetas, training_inst)
    return 0
