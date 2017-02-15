import numpy as np
from scipy.stats import normaltest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from HW2.Utilities import get_x_and_y_datasets, separate_classes


def test_normality():
    first_data, second_data, first_classes, second_classes = separate_classes()
    results = normaltest(first_data, axis=0)
    avg = results[1].mean()
    res_str = "Data {0} normally distributed with average p-value of {1}"
    res_str = res_str.format("is" if avg >= 0.05 else "is not", avg)
    print(res_str)


def lda(num_iterations=20):
    data, classes = get_x_and_y_datasets()
    scores = []
    classifier = LinearDiscriminantAnalysis()

    for i in range(0, num_iterations):
        x_train, x_test, y_train, y_test = train_test_split(data, classes, test_size=(1 - 13000 / 19020),
                                                            random_state=42)
        classifier.fit(x_train, y_train)
        prediction = classifier.predict(x_test)
        scores.append(accuracy_score(y_test, prediction))

    avg_score = np.array(scores).mean()
    print("The success rate with LDA after {0} iterations was {1}%".format(num_iterations, avg_score * 100))


if __name__ == '__main__':
    test_normality()
    lda(num_iterations=100)
