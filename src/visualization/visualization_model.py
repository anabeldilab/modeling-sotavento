import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path


def plot_residuals_vs_fitted(model, model_name, name):
    # Residuals vs Fitted
    fitted_vals = model.predict()
    resids = model.resid
    sns.residplot(
        x=fitted_vals, y=resids, lowess=True, line_kws={"color": "red", "lw": 1}
    )
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title(f"Residuals vs. Fitted Values for {name}")
    output_path = Path(f"graphs/{model_name}/{name}/residuals_vs_fitted.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_qq_plot(model, model_name, name):
    # Residuals vs Fitted
    fitted_vals = model.predict()
    resids = model.resid
    fig = sm.qqplot(resids, fit=True, line="45")
    plt.title(f"Q-Q Plot of Residuals for {name}")
    output_path = Path(f"graphs/{model_name}/{name}/qq_plot.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def calculate_vif(model, model_name, name):
    variables = model.model.exog
    vif = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
    print(vif)
    # Write the VIF to a file
    output_path = Path(f"graphs/{model_name}/{name}/vif.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for i in range(len(vif)):
            f.write(f"{model.model.exog_names[i]}: {vif[i]}\n")


def write_summary_to_file(model, model_name, name):
    output_path = Path(f"graphs/{model_name}/{name}/summary.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(str(model.summary()))


def write_cv_results_to_file(cv_results, model_name, name):
    output_path = Path(f"graphs/models/{model_name}/{name}/cv_results.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for metric_name, metric_values in cv_results.items():
            if metric_name.startswith('test_'):
                # Formatear el nombre de la métrica para mejorar la legibilidad
                formatted_name = metric_name[5:].replace('_', ' ').capitalize()
                # Calcular promedio y desviación estándar
                mean_value = np.mean(metric_values)
                std_dev = np.std(metric_values)
                # Ajustar el formato del mensaje según el tipo de métrica
                if "mape" in metric_name.lower() or "smape" in metric_name.lower():
                    f.write(f"{formatted_name}: {mean_value:.2f}% (+-{std_dev:.2f}%)\n")
                else:
                    f.write(f"{formatted_name}: {mean_value:.4f} (+-{std_dev:.4f})\n")


def predictions_vs_actuals(model, model_name, name, X, y):
    # Predictions vs Actuals
    predictions = model.predict(X)
    plt.scatter(y, predictions)
    plt.xlabel("Actual values")
    plt.ylabel("Predictions")
    plt.title(f"Predictions vs. Actuals for {name}")
    output_path = Path(f"graphs/models/{model_name}/{name}/predictions_vs_actuals.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
