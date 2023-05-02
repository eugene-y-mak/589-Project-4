import numpy as np
import NN


def run_example(reg_lambda, structure, thetas, trainings):
    NN.cost(reg_lambda, structure, thetas, trainings)
    return 0


def main():
    # -----------------Q1-------------------------
    # each row is a neuron, with its weight and bias
    # num of rows = num of neurons in next layer
    # bias term, then weight
    theta_1 = np.array([[0.40000, 0.10000],
                        [0.30000, 0.20000]])
    theta_2 = np.array([[0.70000, 0.50000, 0.60000]])
    run_example(0.0, len([1, 2, 1]), [theta_1, theta_2],
                [{"x": [0.13000], "y": [0.90000]}, {"x": [0.42000], "y": [0.23000]}])
    # -----------------Q2------------------------

    return 0


if __name__ == "__main__":
    main()
