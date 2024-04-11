from models.linear_regression import linear_regression_datasets
from models.polynomial_regression import polynomial_regression_datasets
from models.svm import svm_datasets
from models.knn import knn_datasets
from models.random_forest import random_forest_datasets
from models.gradient_boosting import gradient_boosting_datasets
from models.xgboost import xgboost_datasets



# SVM

# Random Forest

# Regresión logística

# Redes neuronales


def main():
    linear_regression_datasets()
    polynomial_regression_datasets()
    svm_datasets()
    knn_datasets()
    random_forest_datasets()
    gradient_boosting_datasets()
    xgboost_datasets()

 

if __name__ == '__main__':
    main()
