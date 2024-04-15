"""
This module contains functions for preprocessing data in a pandas DataFrame.

Functions:
- treat_nan: Treats NaN values in a DataFrame by removing columns and rows with excessive NaN 
values and imputing the remaining NaN values.
- drop_null_if_above_percentage: Removes rows or columns from a DataFrame if the percentage of null
 values exceeds a specified threshold.
- impute_nan: Imputes NaN values in a DataFrame using different strategies for numeric and 
non-numeric columns.
- normalize_data_zscore: Normalizes data in a DataFrame using the Z-score.
- drop_meaningless_columns: Removes unnecessary columns from a DataFrame.
- make_datetime_index: Transforms the 'data' column of a DataFrame into a datetime type index.
- save_processed_data: Saves a pandas DataFrame to a CSV file in the processed data directory.
- outlier_detection: Detects outliers in a DataFrame using the Z-score method.
- outlier_treatment: Treats outliers in a DataFrame using the Z-score method to identify and impute
them.
- correlation_matrix: Creates and saves a correlation matrix of a pandas DataFrame.
- process_data: Processes a raw text file by loading, converting, modifying, treating NaN values, 
and normalizing the data.
"""

from pathlib import Path
import concurrent.futures
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from src.data_loading import load_and_convert_to_csv
from src.visualization.visualization_preprocessing import (
    plot_dataset_boxplot,
    plot_correlation_matrix,
    plot_dataset_features,
)
from src.visualization.visualization_preprocessing import plot_daily_nulls_per_column


def treat_nan(
    data_frame,
    max_null_column_pct=80,
    max_null_row_pct=1,
    numeric_strategy="mean",
    non_numeric_strategy="most_frequent",
):
    """
    Treats NaN values in a DataFrame, removing columns with more than max_null_column_pct% of NaN,
    rows with more than max_null_row_pct% of NaN, and imputing the remaining NaN values.

    Args:
        data_frame: pandas DataFrame.
        max_null_column_pct: Maximum percentage of null values allowed in a column.
        max_null_row_pct: Maximum percentage of null values allowed in a row.
        numeric_strategy: Imputation strategy for numeric columns ('mean' or 'median').
        non_numeric_strategy: Imputation strategy for non-numeric columns ('most_frequent' or
        'constant').
    """
    print(
        f"Treating NaN in DataFrame:\n"
        f"Rows: {data_frame.shape[0]}, Columns: {data_frame.shape[1]}"
    )
    print(f"Maximum NaN percentage per column: {max_null_column_pct}%")
    data_frame = drop_null_if_above_percentage(
        data_frame, axis=1, max_null_pct=max_null_column_pct
    )
    print(f"Maximum NaN percentage per row: {max_null_row_pct}%")
    data_frame = drop_null_if_above_percentage(
        data_frame, axis=0, max_null_pct=max_null_row_pct
    )
    # Impute NaN with the mean if numeric
    if data_frame.isnull().sum().sum() > 0:
        data_frame = impute_nan(
            data_frame,
            numeric_strategy=numeric_strategy,
            non_numeric_strategy=non_numeric_strategy,
        )
    return data_frame


def drop_null_if_above_percentage(data_frame, axis=0, max_null_pct=50):
    """
    Removes rows or columns from a DataFrame if the percentage of null values exceeds the specified
    threshold.

    Args:
        data_frame: pandas DataFrame.
        axis: 0 for rows, 1 for columns.
        max_null_pct: Threshold percentage of maximum null values allowed.
    """
    # Calculate the minimum number of NON-null values needed not to be dropped,
    # based on the maximum percentage of null values allowed.
    if axis == 1:  # For columns
        threshold = int((100 - max_null_pct) / 100 * len(data_frame))
    else:  # For rows
        threshold = int((100 - max_null_pct) / 100 * len(data_frame.columns))

    return data_frame.dropna(axis=axis, thresh=threshold)


def impute_nan(
    data_frame,
    numeric_strategy="mean",
    non_numeric_strategy="most_frequent",
    interpolation_method=None,
):
    """
    Imputes NaN values in a DataFrame using different strategies for numeric and non-numeric 
    columns.

    Args:
        data_frame: pandas DataFrame.
        numeric_strategy: Imputation strategy for numeric columns ('mean' or 'median').
        non_numeric_strategy: Imputation strategy for non-numeric columns ('most_frequent' or 
        'constant').
    """
    print(
        f"Imputing NaN with strategies: "
        f"numeric={numeric_strategy}, non-numeric={non_numeric_strategy}"
    )
    data_frame_copy = data_frame.copy()

    numeric_cols = data_frame_copy.select_dtypes(include=["number"]).columns

    # Imputation for numeric columns
    if numeric_strategy in ["mean", "median"]:
        numeric_imputer = SimpleImputer(strategy=numeric_strategy)
        data_frame_copy[numeric_cols] = numeric_imputer.fit_transform(
            data_frame_copy[numeric_cols]
        )
    elif numeric_strategy == "interpolate":
        # Ensure the interpolation method is provided
        if not interpolation_method:
            raise ValueError(
                "Interpolation method must be specified when numeric_strategy is 'interpolate'."
            )
        data_frame_copy[numeric_cols] = data_frame_copy[numeric_cols].apply(
            lambda x: x.interpolate(method=interpolation_method)
        )

    # Imputation for non-numeric columns using 'most_frequent'
    non_numeric_cols = data_frame_copy.select_dtypes(exclude=["number"]).columns
    if non_numeric_strategy == "most_frequent" and len(non_numeric_cols) > 0:
        non_numeric_imputer = SimpleImputer(strategy="most_frequent")
        data_frame_copy[non_numeric_cols] = non_numeric_imputer.fit_transform(
            data_frame_copy[non_numeric_cols]
        )

    # Return the DataFrame with imputed values
    return pd.DataFrame(data_frame_copy, columns=data_frame.columns)


def normalize_data_zscore(data_frame):
    """
    Normalizes data in a DataFrame using the Z-score.

    Args:
        data_frame (pd.DataFrame): The pandas DataFrame to normalize.
    """
    data_frame_copy = data_frame.copy()
    numeric_cols = data_frame_copy.select_dtypes(include=["number"]).columns
    data_frame_copy[numeric_cols] = (
        data_frame_copy[numeric_cols] - data_frame_copy[numeric_cols].mean()
    ) / data_frame_copy[numeric_cols].std()
    return data_frame_copy


def drop_meaningless_columns(data_frame):
    """
    Removes unnecessary columns from a DataFrame: 'id', 'id_anemo' and 'n_datos'.

    Args:
        data_frame (pd.DataFrame): The pandas DataFrame to modify
    """
    data_frame_copy = data_frame.copy()
    if "id" in data_frame_copy.columns:
        data_frame_copy.drop(columns=["id"], inplace=True)
    if "id_anemo" in data_frame_copy.columns:
        data_frame_copy.drop(columns=["id_anemo"], inplace=True)
    if "n_datos" in data_frame_copy.columns:
        data_frame_copy.drop(columns=["n_datos"], inplace=True)
    return data_frame_copy


def make_datetime_index(data_frame):
    """
    Transforms the 'data' column of a DataFrame into a datetime type index.

    Args:
        data_frame (pd.DataFrame): The pandas DataFrame to modify
    """
    data_frame_copy = data_frame.copy()
    data_frame_copy["data"] = pd.to_datetime(data_frame_copy["data"], format="%d/%m/%Y %H:%M:%S")
    data_frame_copy.set_index("data", inplace=True)
    return data_frame_copy


def modify_dataset_columns(data_frame):
    """
    Makes specific modifications to the columns of a pandas DataFrame.
    This process includes two main steps: removing unnecessary columns, such as identifiers,
    and transforming the 'data' column into a datetime type index.

    Args:
        data_frame (pd.DataFrame): The pandas DataFrame to modify. This function
        assumes that `data_frame` is mutable and thus is modified directly
        without returning a copy or a modified version of it.
    """
    data_frame_copy = data_frame.copy()
    data_frame_copy = drop_meaningless_columns(data_frame_copy)
    data_frame_copy = make_datetime_index(data_frame_copy)
    return data_frame_copy


def save_processed_data(data, name, processed_data_directory_path):
    """
    Saves a pandas DataFrame to a CSV file in the processed data directory.

    Args:
        data: pandas DataFrame.
        name: Base name of the CSV file.
        processed_data_directory_path: Path of the processed data directory.
    """
    output_path = processed_data_directory_path / f"{name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


def outlier_detection(data_frame, threshold=3):
    """
    Detects outliers in a DataFrame using the Z-score method.

    Args:
        data_frame: pandas DataFrame.
        threshold: Z-score threshold to consider a value as an outlier.
    """
    # Calculate the Z-score on a daily basis for each column (Row is every 10 minutes)
    # Group by day and calculate mean and standard deviation
    daily_z_scores = data_frame.groupby(data_frame.index.date).transform(
        lambda x: (x - x.mean()) / x.std()
    )
    # Calculate outliers based on the Z-score
    return daily_z_scores.abs() > threshold


def outlier_treatment(
    data_frame,
    threshold=3,
    numeric_strategy="interpolate",
    interpolation_method="linear",
):
    """
    Treats outliers in a DataFrame using the Z-score method to identify them
    and then imputes those outliers with the specified strategy for numeric columns.

    Args:
        data_frame: pandas DataFrame.
        threshold: Z-score threshold to consider a value as an outlier.
        numeric_strategy: Imputation strategy for numeric columns ('mean' or 'median').
    """
    # Identify outliers
    outliers = outlier_detection(data_frame, threshold=threshold)
    print(f"Number of outliers detected: {outliers.sum().sum()}")

    # Mark outliers as NaN
    data_frame[outliers] = np.nan

    plot_daily_nulls_per_column(data_frame, "outliers")

    # Impute NaNs (formerly outliers) with the specified strategy
    return impute_nan(
        data_frame,
        numeric_strategy=numeric_strategy,
        non_numeric_strategy=None,
        interpolation_method=interpolation_method,
    )


def perform_correlation_matrix(data_frame, name):
    """
    Creates and saves a correlation matrix of a pandas DataFrame.

    Args:
        data_frame: pandas DataFrame.
        name: Base name of the file.
    """
    correlation_matrix = data_frame.corr()
    plot_correlation_matrix(correlation_matrix, name)


def process_data(raw_text_file_path):
    """
    Processes a raw text file, performing the following operations:
    - Loads and converts the file to a pandas DataFrame.
    - Modifies the DataFrame's columns.
    - Treats NaN values.
    - Normalizes the data using the Z-score.

    Args:
        raw_text_file_path: Path to the raw text file.
    """
    name, data = load_and_convert_to_csv(raw_text_file_path)
    data = modify_dataset_columns(data)
    data = treat_nan(data, max_null_column_pct=90, max_null_row_pct=0)
    data = normalize_data_zscore(data)
    data = outlier_treatment(
        data, threshold=3, numeric_strategy="interpolate", interpolation_method="linear"
    )
    return data, name


def plot_data(data, name):
    """
    Generates and saves visualizations of a dataset.

    Args:
        data: pandas DataFrame.
        name: Base name of the file.
    """
    #plot_dataset_boxplot(data, name)
    #plot_dataset_features(data, name)
    # plot_hourly_feature_per_day(data, name)
    # plot_dataset_boxplot_by_day(data, name)


def join_radiation_and_ac_data():
    """
    Joins the radiation and AC data, aligning the timestamps.
    """
    ac_data = pd.read_csv("data/processed/Analizador AC Fotovoltaica Este (A11).csv")
    radiation_data = pd.read_csv("data/processed/Radiacion Fotovoltaica Este (R1).csv")
    # Quitar valor_max,valor_min y cambiar el nombre de valor_med a radiación
    radiation_data.drop(columns=["valor_max", "valor_min"], inplace=True)
    radiation_data.rename(columns={"valor_med": "radiation"}, inplace=True)
    joined_data = ac_data.join(radiation_data, how="outer")
    joined_data.to_csv("data/processed/Analizador AC Fotovoltaica Este (A11) + Radiacion Fotovoltaica Este (R1).csv", index=False)


def main():
    """
    Main function of the script. Processes raw data files,
    saves processed data, and generates visualizations of the data.
    """
    raw_data_directory_path = Path("data/raw")
    processed_data_directory_path = Path("data/processed")

    data_names = []
    for file_path in raw_data_directory_path.glob("*.txt"):
        data, name = process_data(file_path)
        perform_correlation_matrix(data, name)
        save_processed_data(data, name, processed_data_directory_path)
        data_names.append((data, name))

    join_radiation_and_ac_data()


    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(plot_data, data, name) for data, name in data_names]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An exception occurred: {e}")


if __name__ == "__main__":
    main()
