import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time

from HW2.Utilities import get_x_and_y_datasets, plotter


def knn_with_test_split(num_iterations=20):
    start = time.time()
    data, classes = get_x_and_y_datasets()

    possible_ks = np.arange(1, 100, step=2)
    scores = np.empty([50, num_iterations])

    for i in range(0, num_iterations):
        x_train, x_test, y_train, y_test = train_test_split(data, classes, test_size=(1 - 13000 / 19020),
                                                            random_state=42)

        for k in possible_ks:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_train, y_train)
            prediction = knn.predict(x_test)
            scores[int(k / 2)][i] = accuracy_score(y_test, prediction)

    mean_errors = [arr.mean() for arr in scores]
    maximum = max(mean_errors)
    index = mean_errors.index(maximum)
    print('The best k with {0} iterations was {1} with a success rate of {2}'.format(num_iterations, possible_ks[index],
                                                                                     max(mean_errors)))

    mis_errors = [1 - x for x in mean_errors]
    plotter(possible_ks, mis_errors, 'K-Neighbors', 'Misclassification Rate', 'KNN with Test Split')
    print("That took {0} seconds".format(time.time() - start))


def knn_with_cross_fold_validation(num_iterations=1):
    data, classes = get_x_and_y_datasets()

    possible_ks = np.arange(1, 100, step=2)
    cv_scores = np.empty([50, num_iterations])

    for i in range(0, num_iterations):
        for k in possible_ks:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, data, classes, cv=10, scoring='accuracy')
            cv_scores[int(k / 2)][i] = (scores.mean())

    mean_cv_errors = [arr.mean() for arr in cv_scores]
    maximum = max(mean_cv_errors)
    index = mean_cv_errors.index(maximum)

    optimal_k = possible_ks[index]
    print('The best k with {0} iterations was {1} with a success rate of {2}'.format(num_iterations, optimal_k,
                                                                                     max(mean_cv_errors)))

    inverse_errors = [1 - x for x in mean_cv_errors]
    plotter(possible_ks, inverse_errors, 'Number of Neighbors K', 'Misclassification Rate', 'KNN with Cross Fold '
                                                                                            'Validation')


if __name__ == '__main__':
    # knn_with_test_split(num_iterations=50)
    knn_with_cross_fold_validation(num_iterations=100)
