import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from warmUpExercise import warm_up_exercise
from plotData import plot_data
from computeCost import compute_cost
from gradientDescent import gradient_descent

# Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     warmUpExercise.py
#     plotData.py
#     gradientDescent.py
#     computeCost.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#


# ==================== Part 1: Basic Function ====================

print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')
print(warm_up_exercise())

input('Program paused. Press <ENTER> to continue.\n')


# ======================= Part 2: Plotting =======================

print('Plotting Data ...\n')

data = np.loadtxt('ex1data1.txt', delimiter=',')
x = data[:, [0]]
y = data[:, [1]]
m = len(y)  # Number of training examples

# Plot Data

plot_data(x, y)

input('Program paused. Press <ENTER> to continue.\n')


# =================== Part 3: Cost and Gradient descent ===================

x = np.concatenate((np.ones([m, 1]), x), axis=1)  # Add a column of ones to x as first column
theta = np.zeros([2, 1])  # Initialize fitting parameters

# Some gradient descent settings
num_iters = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')
# Compute and display initial cost
J = compute_cost(x, y, theta)
print('With theta = [0  0]\nCost computed =', J[0][0], '\n')
print('Expected cost value (approx) 32.07\n')

# Further testing of the cost function
theta = np.array([[-1], [2]])
J = compute_cost(x, y, theta)
print('\nWith theta = [-1  2]\nCost computed =', J[0][0], '\n')
print('Expected cost value (approx) 54.24\n')


input('Program paused. Press enter to continue.\n')

print('\nRunning Gradient Descent ...\n')
# Run gradient descent
theta, j_hist = gradient_descent(x, y, theta, alpha, num_iters)

# Print theta to screen
print('Theta found by gradient descent:\n')
print(theta[0][0])
print(theta[1][0])
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

# Plot the linear fit

plt.plot(x[:, [1]], np.dot(x, theta), '-')
plt.legend(('Training data', 'Linear regression'))

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(np.array([1, 3.5]), theta)
print('For population = 35,000, we predict a profit of', predict1[0] * 10000)
predict2 = np.dot(np.array([1, 7]), theta)
print('For population = 70,000, we predict a profit of', predict2[0] * 10000)

input('Program paused. Press enter to continue.\n')


# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, num=100)
theta1_vals = np.linspace(-1, 4, num=100)

# Initialize t and j_vals to a matrix of 0's
j_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out j_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        j_vals[i, j] = compute_cost(x, y, t)

# Surface plot

fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, j_vals.T, cmap=cm.coolwarm)
ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')

# Contour plot

# Plot j_vals as 15 contours spaced logarithmically between 0.01 and 100

fig2 = plt.figure()
plt.contour(theta0_vals, theta1_vals, j_vals.T, np.logspace(-2, 3, num=20))
plt.xlabel('theta_0')
plt.ylabel('theta_1')
