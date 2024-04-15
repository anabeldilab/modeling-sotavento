from pathlib import Path
import joblib
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
from src.visualization.visualization_model import (
    predictions_vs_actuals,
    plot_residuals_vs_fitted,
)
from src.utils.cross_validation import perform_regression_cv


def perform_svr(csv_file, name, dependent_variable, independent_variables):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Select the dependent and independent variables
    X = df[independent_variables]  # Independent variables
    y = df[dependent_variable]  # Dependent variable

    # It's often useful to scale the features when using SVM
    model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

    # Cross-validation
    perform_regression_cv(model, X, y, folds=5, name=name, model_name="svm")

    model.fit(X, y)

    predictions_vs_actuals(model, "svm", name, X, y)

    # Residuals vs Fitted
    # plot_residuals_vs_fitted(model, "svm", name)

    output_path = Path("models/svm/", "svm_model.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def load_svm_model():
    model = joblib.load("models/svm/svm_model.pkl")
    return model


def svm_datasets():
    perform_svr(
        csv_file="data/processed/Analizador AC Fotovoltaica Este (A11) + Radiacion Fotovoltaica Este (R1).csv",
        name="Analizador AC Fotovoltaica Sur (A12) + Radiacion Este (R1)",
        dependent_variable="W",
        independent_variables=["radiation"],
    )
