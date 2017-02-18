import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB


def get_data():
    return np.array([[.6585, .2444], [2.246, .5281], [-2.7665, -3.8303],
                     [-1.2565, 3.4912], [-.7973, 1.2288], [1.117, 2.2637]])


def get_classes():
    return np.array([0, 0, 0, 1, 1, 1])


def qda_test():
    data = get_data()
    classes = get_classes()

    classifier = QuadraticDiscriminantAnalysis()
    classifier.fit(data, classes)
    prediction = classifier.predict([[0, 1]])
    probabilities = classifier.predict_proba([[0, 1]])
    return prediction, probabilities


def naive_bayes_test():
    data = get_data()
    classes = get_classes()

    classifier = GaussianNB()
    classifier.fit(data, classes)
    prediction = classifier.predict([[0, 1]])
    probabilities = classifier.predict_proba([[0, 1]])
    return prediction, probabilities


if __name__ == '__main__':
    q_predict, q_probs = qda_test()
    n_predict, n_probs = naive_bayes_test()

    print("Class {0} was predicted with QDA with a posterior probability of {1}".format(q_predict[0],
                                                                                        q_probs[0][q_predict[0]]))
    print("Class {0} was predicted with Naive Bayes with a posterior probability of {1}".format(n_predict[0],
                                                                                                n_probs[0][
                                                                                                    n_predict[0]]))

    # Class 1 was predicted with QDA with a posterior probability of 0.9932105479042859
    # Class 1 was predicted with Naive Bayes with a posterior probability of 0.713965862288279
