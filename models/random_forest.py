from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.visualization.visualization_model import (
    predictions_vs_actuals,
    plot_residuals_vs_fitted,
)
from src.utils.cross_validation import perform_regression_cv

MODEL_NAME = "random-forest"


def perform_random_forest(
    csv_file, name, dependent_variable, independent_variables
):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Select the dependent and independent variables
    X = df[independent_variables]  # Independent variables
    y = df[dependent_variable]  # Dependent variable

    # Perform the linear regression
    model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100))

    perform_regression_cv(model, X, y, folds=5, name=name, model_name=MODEL_NAME)

    model.fit(X, y)

    predictions_vs_actuals(model, MODEL_NAME, name, X, y)

    #plot_residuals_vs_fitted(model, MODEL_NAME, name)

    output_path = Path("models/random_forest/", 'random_forest_model.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)



def random_forest_datasets():
    perform_random_forest(
        csv_file="data/processed/Analizador AC Fotovoltaica Este (A11) + Radiacion Fotovoltaica Este (R1).csv",
        name="Analizador AC Fotovoltaica Sur (A12) + Radiacion Este (R1)",
        dependent_variable="W",
        independent_variables=["radiation"],
    )