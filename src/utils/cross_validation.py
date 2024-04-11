import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    make_scorer,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from src.visualization.visualization_model import write_cv_results_to_file


def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))
    )


def perform_regression_cv(model, X, y, folds=5, name="", model_name=""):
    # Cross validation
    # Metrics definition

    smape_scorer = make_scorer(smape, greater_is_better=False)  # Make smape compatible

    scoring = {
        "MSE": make_scorer(mean_squared_error, greater_is_better=False),
        "RMSE": make_scorer(
            lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            greater_is_better=False,
        ),
        "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
        "MAPE": make_scorer(mean_absolute_percentage_error, greater_is_better=False),
        "SMAPE": smape_scorer,
        "R2": "r2",
    }

    cv_results = cross_validate(model, X, y, cv=folds, scoring=scoring)

    write_cv_results_to_file(cv_results, model_name, name)
    
