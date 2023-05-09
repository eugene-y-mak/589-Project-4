import sys

import numpy as np
import pandas as pd

import NN

EXAMPLE = 1


def run_example(reg_lambda, structure, thetas, trainings):
    NN.cost(reg_lambda, structure, thetas, trainings, "x", "y", True)
    NN.back_propagation(alpha=1 / (10 ** 3), reg_lambda=reg_lambda, num_layers=structure, thetas=thetas,
                        trainings=trainings, input_label="x", output_label="y", do_print=True)

    # uncomment here to do numerical gradient check
    # NN.numerical_gradients_check(thetas, 0.1, structure, trainings, "x", "y", False)
    # NN.numerical_gradients_check(thetas, 0.000001, structure, trainings, "x", "y", False)

    # test of convergence
    # NN.train_NN(1/(10**3), 10e-8, reg_lambda, structure, thetas, trainings, "x", "y")
    return 0


def main(example):
    if example == 1:
        print("TESTING EXAMPLE 1")
        # -----------------Q1-------------------------
        # For thetas:
        # num of rows = num of neurons in next layer
        # first col is bias terms, then weights
        data = [{"x": [0.13000], "y": [0.90000]}, {"x": [0.42000], "y": [0.23000]}]
        df = pd.DataFrame(data)

        run_example(0.0, len([1, 2, 1]),
                    [np.array([[0.40000, 0.10000], [0.30000, 0.20000]]),  # theta 1
                     np.array([[0.70000, 0.50000, 0.60000]])],  # theta 2
                    df)

    elif example == 2:
        print("TESTING EXAMPLE 2")
        # -----------------Q2------------------------
        run_example(0.250, len([2, 4, 3, 2]),
                    [np.array([[0.42000, 0.15000, 0.40000],  # theta 1
                               [0.72000, 0.10000, 0.54000],
                               [0.01000, 0.19000, 0.42000],
                               [0.30000, 0.35000, 0.68000]]),
                     np.array([[0.21000, 0.67000, 0.14000, 0.96000, 0.87000],  # theta 2
                               [0.87000, 0.42000, 0.20000, 0.32000, 0.89000],
                               [0.03000, 0.56000, 0.80000, 0.69000, 0.09000]]),
                     np.array([[0.04000, 0.87000, 0.42000, 0.53000],  # theta 3
                               [0.17000, 0.10000, 0.95000, 0.69000]])],
                    pd.DataFrame([{"x": [0.32000, 0.68000], "y": [0.75000, 0.98000]},
                     {"x": [0.83000, 0.02000], "y": [0.75000, 0.28000]}]))
    return 0


if __name__ == "__main__":
    main(EXAMPLE)
