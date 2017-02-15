import random

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import normaltest

FILE_NAME = 'magic04.csv'


def get_x_and_y_datasets():
    arr = np.arange(0, 10)

    data = np.loadtxt(FILE_NAME, delimiter=',', usecols=arr)
    classes = np.loadtxt(FILE_NAME, delimiter=',', usecols=[10], dtype=str)
    classes = [x[2] for x in classes]

    return data, classes


def test_normality():
    # first_data, second_data, first_classes, second_classes = separate_classes()
    data, classes = get_x_and_y_datasets()
    results = normaltest(data, axis=0)
    avg = results[1].mean()
    res_str = "Data {0} normally distributed with average p-value of {1}"
    res_str = res_str.format("is" if avg >= 0.05 else "is not", avg)
    print(res_str)


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
    plt.savefig("/home/tesla/PycharmProjects/474-HW/HW2/{0}".format(title))
    # plt.show()


if __name__ == '__main__':
    test_normality()
