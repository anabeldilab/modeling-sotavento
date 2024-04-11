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


def polynomial_regression_datasets():
    perform_polynomial_regression(
        csv_file="data/processed/Analizador AC Fotovoltaica Este (A11).csv",
        name="Analizador AC Fotovoltaica Este (A11)",
        dependent_variable="Wh_e",
        independent_variables=["V", "I", "W", "VAr"],
        degree=2,
    )

    perform_polynomial_regression(
        csv_file="data/processed/Analizador AC Fotovoltaica Oeste (A13).csv",
        name="Analizador AC Fotovoltaica Oeste (A13)",
        dependent_variable="W",
        independent_variables=["V", "I", "W", "VAr"],
        degree=2,
    )

    perform_polynomial_regression(
        csv_file="data/processed/Analizador AC Fotovoltaica Sur (A12).csv",
        name="Analizador AC Fotovoltaica Sur (A12)",
        dependent_variable="W",
        independent_variables=["V", "I", "W", "VAr"],
        degree=2,
    )
