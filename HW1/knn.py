from collections import Counter

import numpy as np


# Takes in two points and returns the distance between them by calculating the norm of their difference
def dist(p1, p2):
    return np.linalg.norm(p1 - p2)


# Finds the most frequently occurring class among the k-closest points to a given point
# Ties are broken arbitrarily, as numpy handles the sorting
def knn(data, classes, point, k):
    # Array of distance between each data point and the given point
    distances = np.array([dist(point, p2) for p2 in data])

    # Sort the indices by the distances so we can properly access their associated classes
    a = distances.argsort()

    # Create a Python Counter object of the classes of the k-closest neighbors
    counter = Counter(classes[a[0:k]])

    # Find the most frequently occurring class among those closest
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

    c = np.array(['Red', 'Red', 'Red', 'Green', 'Green', 'Red'])

    p = np.array([0, 0, 0], np.float32)

    test_k = int(input("Enter a number for k: "))

    prediction = knn(d, c, p, test_k)

    print prediction
