import numpy as np
from matplotlib import pyplot as plt

FILE_NAME = 'magic04.csv'


def get_x_and_y_datasets():
    arr = np.arange(0, 10)
    data = np.loadtxt(FILE_NAME, delimiter=',', usecols=arr)
    classes = np.loadtxt(FILE_NAME, delimiter=',', usecols=[10], dtype=str)
    classes = [x[2] for x in classes]

    return data, classes

    # DELETE ME
    lower_upper = 1000
    upper_upper = 18020
    first_data = data[0:lower_upper]
    last_data = data[upper_upper:]
    first_classes = classes[0:lower_upper]
    last_classes = classes[upper_upper:]
    data = np.vstack((first_data, last_data))
    classes = np.append(first_classes, last_classes)
    # END DELETE

    return data, classes


def plotter(x_set, y_set, x_lab, y_lab, title):
    plt.plot(x_set, y_set)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    plt.show()
