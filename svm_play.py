from sklearn import svm
import numpy as np
from numpy import random
import matplotlib.pyplot as plt


def svm_driver():
    data_1 = random.randn(100, 2)
    data_2 = data_1 * 3 + 5

    classes = np.append(np.zeros((1, 100)), np.ones((1, 100)))
    # print(classes)

    all_data = np.vstack((data_1, data_2))
    classifier = svm.SVC()
    classifier.fit(all_data, classes)

    print(classifier)


    # plt.plot(data_1[:, 0], data_1[:, 1], 'ro')
    # plt.plot(data_2[:, 0], data_2[:, 1], 'bo')
    # plt.show()


if __name__ == '__main__':
    svm_driver()
