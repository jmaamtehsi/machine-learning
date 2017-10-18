import numpy as np
from computeCost import compute_cost


def gradient_descent(x, y, theta, alpha, num_iters):
    """Performs gradient descent to learn theta. Updates theta by taking num_iters
    gradient steps with learning rate alpha."""

    m = len(y)  # Number of training examples
    j_hist = np.zeros([num_iters, 1])

    for ind in range(num_iters):

        h_minus_y = np.dot(x, theta) - y
        theta = theta - (alpha / m) * np.dot(x.T, h_minus_y)

        j_hist[ind] = compute_cost(x, y, theta)

    return theta, j_hist
