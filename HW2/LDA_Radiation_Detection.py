import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from HW2.Utilities import get_x_and_y_datasets


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
