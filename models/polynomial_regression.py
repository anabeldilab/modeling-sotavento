from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.visualization.visualization_model import (
    predictions_vs_actuals,
    plot_residuals_vs_fitted,
)
from src.utils.cross_validation import perform_regression_cv


def perform_polynomial_regression(
    csv_file, name, dependent_variable, independent_variables, degree=2
):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Select the dependent and independent variables
    X = df[independent_variables]  # Independent variables
    y = df[dependent_variable]  # Dependent variable

    model = make_pipeline(
        PolynomialFeatures(degree, include_bias=False),
        StandardScaler(),
        LinearRegression(),
    )

    perform_regression_cv(
        model, X, y, folds=5, name=name, model_name="polynomial-regression"
    )

    model.fit(X, y)

    predictions_vs_actuals(model, "polynomial-regression", name, X, y)

    #plot_residuals_vs_fitted(model, "polynomial-regression", name)

    output_path = Path("models/polynomial_regression/", 'polynomial_regression_model.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def polynomial_regression_datasets():
    perform_polynomial_regression(
        csv_file="data/processed/Analizador AC Fotovoltaica Este (A11) + Radiacion Fotovoltaica Este (R1).csv",
        name="Analizador AC Fotovoltaica Sur (A12) + Radiacion Este (R1)",
        dependent_variable="W",
        independent_variables=["radiation"],
    )
