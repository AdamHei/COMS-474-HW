import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

FILE_NAME = 'magic04.csv'


def get_x_and_y_datasets():
    arr = np.arange(0, 10)
    data = np.loadtxt('magic04.csv', delimiter=',', usecols=arr)
    classes = np.loadtxt('magic04.csv', delimiter=',', usecols=[10], dtype=str)
    classes = [x[2] for x in classes]
    return data, classes


def test_knn():
    start = time.time()
    data, classes = get_x_and_y_datasets()

    # DELETE ME
    lower_upper = 2000
    upper_upper = 17000
    first_data = data[0:lower_upper]
    last_data = data[upper_upper:]
    first_classes = classes[0:lower_upper]
    last_classes = classes[upper_upper:]
    data = np.vstack((first_data, last_data))
    classes = np.append(first_classes, last_classes)
    # END DELETE

    x_train, x_test, y_train, y_test = train_test_split(data, classes, test_size=(1 - 13000 / 19020), random_state=42)

    possible_ks = np.arange(1, 100, step=2)
    cv_scores = []

    for k in possible_ks:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())

    MSE = [1 - x for x in cv_scores]

    optimal_k = possible_ks[cv_scores.index(max(cv_scores))]
    print(optimal_k)
    print(max(cv_scores))

    plt.plot(possible_ks, MSE)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Misclassification Error')
    plt.show()

    print("That took {0} seconds".format(time.time() - start))


if __name__ == '__main__':
    test_knn()
