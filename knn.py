from collections import Counter

import numpy as np


def dist(p1, p2):
    return np.linalg.norm(p1 - p2)


def knn(data, classes, point, k):
    distances = np.array([dist(point, p2) for p2 in data])

    a = distances.argsort()

    counter = Counter(classes[a[0:k]])

    majority = counter.most_common(1)

    return majority[0][0]


if __name__ == '__main__':
    d = np.array([
        [2, 3, 0],
        [2, 0, 1],
        [0, 1, 3],
        [0, 1, 2],
        [-1, 0, 1],
        [1, -1, 1]
    ], np.float32)

    c = np.array(['R', 'R', 'R', 'G', 'G', 'R'])

    p = np.array([0, 0, 0], np.float32)

    prediction = knn(d, c, p, 1)

    print prediction
