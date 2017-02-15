import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

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


def knn_with_test_split(num_iterations=20):
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
    plt.plot(possible_ks, mis_errors)
    plt.xlabel('K-Neighbors')
    plt.ylabel('Misclassification Rate')
    plt.title('KNN with Test Split')
    plt.show()


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
    plt.plot(possible_ks, inverse_errors)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Misclassification Error')
    plt.title('KNN with Cross Fold Validation')
    plt.show()


if __name__ == '__main__':
    knn_with_test_split(num_iterations=5)
    # knn_with_cross_fold_validation()
