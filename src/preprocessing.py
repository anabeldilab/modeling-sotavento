import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from visualization import load_and_convert_to_csv, create_nan_csv, plot_dataset_features
from visualization import plot_dataset_boxplot, plot_hourly_feature_per_day, plot_correlation_matrix


def treat_nan(data_frame):
    # Eliminar columnas con más de 80% de NaN
    data_frame = drop_null_if_above_percentage(data_frame, axis=1, max_null_pct=80)
    # Eliminar filas con más de 1% de NaN
    data_frame = drop_null_if_above_percentage(data_frame, axis=0, max_null_pct=1)
    # Imputar NaN con la media si es numérico
    data_frame = impute_nan(data_frame, 'mean', 'most_frequent')
    return data_frame


def drop_null_if_above_percentage(data_frame, axis=0, max_null_pct=50):
    """
    Elimina filas o columnas de un DataFrame si el porcentaje de valores nulos es mayor que (100 - max_null_pct).

    Parámetros:
    - data_frame: DataFrame de pandas.
    - axis: 0 para filas, 1 para columnas.
    - max_null_pct: Porcentaje máximo de valores nulos permitidos para conservar la fila/columna.
    """
    # Calcular el número mínimo de valores NO nulos necesarios para no ser eliminado,
    # basado en el porcentaje máximo de valores nulos permitidos.
    if axis == 1:  # Para columnas
        threshold = int((100 - max_null_pct) / 100 * len(data_frame))
    else:  # Para filas
        threshold = int((100 - max_null_pct) / 100 * len(data_frame.columns))
    
    return data_frame.dropna(axis=axis, thresh=threshold)



def impute_nan(data_frame, numeric_strategy='mean', non_numeric_strategy='most_frequent'):
    data_frame_copy = data_frame.copy()

    # Imputación para columnas numéricas
    if numeric_strategy in ['mean', 'median']:
        numeric_cols = data_frame_copy.select_dtypes(include=['number']).columns
        numeric_imputer = SimpleImputer(strategy=numeric_strategy)
        data_frame_copy[numeric_cols] = numeric_imputer.fit_transform(data_frame_copy[numeric_cols])

    # Imputación para columnas no numéricas utilizando 'most_frequent'
    non_numeric_cols = data_frame_copy.select_dtypes(exclude=['number']).columns
    if non_numeric_strategy == 'most_frequent' and len(non_numeric_cols) > 0:
        non_numeric_imputer = SimpleImputer(strategy='most_frequent')
        data_frame_copy[non_numeric_cols] = non_numeric_imputer.fit_transform(data_frame_copy[non_numeric_cols])

    # Retorna el DataFrame con los valores imputados
    return pd.DataFrame(data_frame_copy, columns=data_frame.columns)


def normalize_data_zscore(data_frame):
    data_frame_copy = data_frame.copy()
    numeric_cols = data_frame_copy.select_dtypes(include=['number']).columns
    data_frame_copy[numeric_cols] = (data_frame_copy[numeric_cols] - data_frame_copy[numeric_cols].mean()) / data_frame_copy[numeric_cols].std()
    return data_frame_copy


def drop_meaningless_columns(data_frame):
    if ("id" in data_frame.columns):
        data_frame.drop(columns=["id"], inplace=True)
    elif ("id_anemo" in data_frame.columns):
        data_frame.drop(columns=["id_anemo"], inplace=True)

def make_datetime_index(data_frame):
    data_frame['data'] = pd.to_datetime(data_frame['data'], format='%d/%m/%Y %H:%M:%S')
    data_frame.set_index('data', inplace=True)

def modify_dataset_columns(data_frame):
    drop_meaningless_columns(data_frame)
    make_datetime_index(data_frame)


def main():
    raw_data_directory_path = Path('data/raw')
    processed_data_directory_path = Path('data/processed')

    for raw_text_file_path in raw_data_directory_path.glob('*.txt'):
        name, data = load_and_convert_to_csv(raw_text_file_path, processed_data_directory_path)

        modify_dataset_columns(data) 
         
        # Mostrar el porcentaje de NaN antes de cualquier tratamiento
        #plot_nan_percentage(data, name)

        # Mostrar la distribución de NaN por hora del día
        #plot_hourly_nulls_per_day(data, 'data', name)

        # Mostrar la distribución de NaN por día
        #plot_daily_nulls_per_column(data, 'data', name)

        create_nan_csv(data, name)
 
        # Tratar eliminación de NaN
        data = drop_null_if_above_percentage(data, axis=1, max_null_pct=80)
        data = drop_null_if_above_percentage(data, axis=0, max_null_pct=0)

        # Tratar imputación de NaN
        #data = treat_nan(data)

        # Normalizar los datos
        #data = normalize_data_zscore(data)

        # See outliers using boxplot
        #plot_dataset_boxplot(data, name)

        #plot_dataset_features(data, name)

        #plot_hourly_feature_per_day(data, 'data', name)

        # Matriz de correlación
        #correlation_matrix = data.corr()

        #plot_correlation_matrix(correlation_matrix, name)
        
        # Guardar el DataFrame tratado en un archivo CSV
        output_path = processed_data_directory_path / f'{name}.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()
