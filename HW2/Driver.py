import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from HW2.KNN_Radiation_Detection import knn_with_test_split, knn_with_cross_fold_validation
from HW2.LDA_Radiation_Detection import lda
from HW2.Naive_Bayes_RD import naive_bayes
from HW2.QDA_Radiation_Detection import qda


if __name__ == '__main__':
    knn_with_test_split(num_iterations=100)
    knn_with_cross_fold_validation(num_iterations=100)
    lda(num_iterations=1000)
    qda(num_iterations=1000)
    naive_bayes(num_iterations=1000)
