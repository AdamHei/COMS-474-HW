import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import normaltest
from sklearn.preprocessing import normalize

FILE_NAME = 'magic04.csv'


###
### Utility function class for use in classifier tests
###

def get_x_and_y_datasets():
    arr = np.arange(0, 10)

    data = np.loadtxt(FILE_NAME, delimiter=',', usecols=arr)
    classes = np.loadtxt(FILE_NAME, delimiter=',', usecols=[10], dtype=str)
    classes = [x[2] for x in classes]

    data = normalize(data, axis=0)

    return data, classes


def test_normality():
    data, classes = get_x_and_y_datasets()
    results = normaltest(data, axis=0)
    avg = results[1].mean()
    res_str = "Data {0} normally distributed with average p-value of {1}"
    res_str = res_str.format("is" if avg >= 0.05 else "is not", avg)
    print(res_str)


def plotter(x_set, y_set, x_lab, y_lab, title, iterations):
    plt.plot(x_set, y_set)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title("{0} with {1} iterations".format(title, iterations))

    plt.savefig("/home/tesla/PycharmProjects/474-HW/HW2/{0}".format(title))
    plt.show()
