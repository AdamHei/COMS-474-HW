from HW2.KNN_Radiation_Detection import knn_with_test_split, knn_with_cross_fold_validation
from HW2.LDA_Radiation_Detection import lda
from HW2.Naive_Bayes_RD import naive_bayes
from HW2.QDA_Radiation_Detection import qda


## Main method where each classifier is tested
from HW2.Utilities import test_normality

if __name__ == '__main__':
    knn_with_test_split(num_iterations=100)
    knn_with_cross_fold_validation(num_iterations=100)

    test_normality()

    lda(num_iterations=100)
    qda(num_iterations=100)
    naive_bayes(num_iterations=100)


    # The success rate with KNN after 100 iterations was ~82%
    # The success rate with LDA after 100 iterations was 78.52516193323372%
    # The success rate with QDA after 100 iterations was 78.40890217571832%
    # The success rate with Naive Bayes after 100 iterations was 72.67895698388973%
