import numpy as np


def compute_cost(x, y, theta):
    """Compute cost for linear regression. Computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y.

    X, y, and theta are np arrays."""

    m = len(y)  # Number of training examples
    h_minus_y = np.dot(x, theta) - y

    return (1/(2 * m)) * np.dot(h_minus_y.T, h_minus_y)
