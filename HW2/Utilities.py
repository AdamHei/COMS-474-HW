import random

import numpy as np
from matplotlib import pyplot as plt

FILE_NAME = 'magic04.csv'


def get_x_and_y_datasets():
    arr = np.arange(0, 10)

    data = np.loadtxt(FILE_NAME, delimiter=',', usecols=arr)
    classes = np.loadtxt(FILE_NAME, delimiter=',', usecols=[10], dtype=str)
    classes = [x[2] for x in classes]

    # shuffle(data, classes)

    return data, classes
    return data[0:2000], classes[0:2000]


def separate_classes():
    arr = np.arange(0, 10)

    data = np.loadtxt(FILE_NAME, delimiter=',', usecols=arr)
    classes = np.loadtxt(FILE_NAME, delimiter=',', usecols=[10], dtype=str)
    classes = [x[2] for x in classes]

    index = 0
    for i in range(0, data.shape[0] - 1):
        if classes[i] != classes[i + 1]:
            index = i

    return data[0:index + 1], data[index + 1:], classes[0:index + 1], classes[index + 1:]


def shuffle(data, classes):
    for i in range(0, data.shape[0]):
        index = random.randint(0, data.shape[0] - 1)

        dtemp = data[index]
        ctemp = classes[index]

        data[index] = data[i]
        classes[index] = classes[i]

        data[i] = dtemp
        classes[i] = ctemp


def plotter(x_set, y_set, x_lab, y_lab, title):
    plt.plot(x_set, y_set)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    separate_classes()
