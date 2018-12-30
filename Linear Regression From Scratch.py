#!python3

import numpy as np
import random
from statistics import mean
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def create_data_set(size, variance, slope):
    # creates data following a line, with a prespecified variation from that line
    # size is the number of points generated, variance is max distance from line, slope is slope of the line
    x = []
    y = []
    for i in range(size):
        x.append(i)
        y.append(i * slope + random.randrange(-variance, variance))

    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    return x, y


def calculate_slope_and_intercept(x, y):
    # calculates line
    slope = (((mean(x) * mean(y)) - mean(x * y)) /
             (mean(x)*mean(x) - mean(x * x)))
    intercept = mean(y) - slope * mean(x)

    return slope, intercept


def coefficient_of_determination(data, line):
    # calculates reliability of calculations
    data_mean = [mean(data) for y in data]

    squared_error_regression = sum((line - data) * (line - data))
    squared_error_mean = sum((data_mean - data) * (data_mean - data))

    print(squared_error_regression)
    print(squared_error_mean)

    r_squared = 1 - (squared_error_regression/squared_error_mean)

    return r_squared


xs, ys = create_data_set(50, 12, 3)
m, b = calculate_slope_and_intercept(xs, ys)
best_fit = [(m*x) + b for x in xs]
plt.scatter(xs, ys, color='r', label='Data')
plt.plot(xs, best_fit, color='c', label='Line of best fit')
plt.legend(loc=4)
r_square = coefficient_of_determination(ys, best_fit)
print('slope: ', m, '\nintercept: ', b, '\nr_squared', r_square)
plt.show()
