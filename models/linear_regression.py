import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.visualization.visualization_model import (
    predictions_vs_actuals,
    plot_residuals_vs_fitted,
)
from src.utils.cross_validation import perform_regression_cv

MODEL_NAME = "linear-regression"


def perform_linear_regression(
    csv_file, name, dependent_variable, independent_variables
):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Select the dependent and independent variables
    X = df[independent_variables]  # Independent variables
    y = df[dependent_variable]  # Dependent variable

    # Perform the linear regression
    model = make_pipeline(StandardScaler(), LinearRegression())

    perform_regression_cv(model, X, y, folds=5, name=name, model_name=MODEL_NAME)

    model.fit(X, y)

    predictions_vs_actuals(model, MODEL_NAME, name, X, y)

    #plot_residuals_vs_fitted(model, MODEL_NAME, name)



def linear_regression_datasets():
    perform_linear_regression(
        csv_file="data/processed/Analizador AC Fotovoltaica Este (A11).csv",
        name="Analizador AC Fotovoltaica Este (A11)",
        dependent_variable="W",
        independent_variables=["V", "I"],
    )

    perform_linear_regression(
        csv_file="data/processed/Analizador AC Fotovoltaica Oeste (A13).csv",
        name="Analizador AC Fotovoltaica Oeste (A13)",
        dependent_variable="W",
        independent_variables=["V", "I"],
    )

    perform_linear_regression(
        csv_file="data/processed/Analizador AC Fotovoltaica Sur (A12).csv",
        name="Analizador AC Fotovoltaica Sur (A12)",
        dependent_variable="W",
        independent_variables=["V", "I"],
    )
