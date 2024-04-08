import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
import concurrent.futures
from src.data_loading import load_and_convert_to_csv, create_nan_csv
from src.visualization import plot_dataset_boxplot, plot_hourly_feature_per_day, plot_correlation_matrix, plot_dataset_features
from src.visualization import plot_dataset_boxplot_by_day, plot_daily_nulls_per_column, plot_hourly_nulls_per_day


def treat_nan(data_frame, max_null_column_pct=80, max_null_row_pct=1, numeric_strategy='mean', non_numeric_strategy='most_frequent'):
    """
    Trata los valores NaN de un DataFrame, eliminando columnas con más de max_null_column_pct% de NaN,
    filas con más de max_null_row_pct% de NaN, e imputando los valores NaN restantes.

    Args:
        data_frame: DataFrame de pandas.
        max_null_column_pct: Porcentaje máximo de valores nulos permitidos en una columna.
        max_null_row_pct: Porcentaje máximo de valores nulos permitidos en una fila.
        numeric_strategy: Estrategia de imputación para columnas numéricas ('mean' o 'median').
        non_numeric_strategy: Estrategia de imputación para columnas no numéricas ('most_frequent' o 'constant').
    """
    print(f'Tratando NaN en DataFrame con {data_frame.shape[0]} filas y {data_frame.shape[1]} columnas.')
    print(f'Porcentaje máximo de NaN por columna: {max_null_column_pct}%')
    data_frame = drop_null_if_above_percentage(data_frame, axis=1, max_null_pct=max_null_column_pct)
    print(f'Porcentaje máximo de NaN por fila: {max_null_row_pct}%')
    data_frame = drop_null_if_above_percentage(data_frame, axis=0, max_null_pct=max_null_row_pct)
    # Imputar NaN con la media si es numérico 
    if data_frame.isnull().sum().sum() > 0:
        data_frame = impute_nan(data_frame, numeric_strategy=numeric_strategy, non_numeric_strategy=non_numeric_strategy)
    return data_frame


def drop_null_if_above_percentage(data_frame, axis=0, max_null_pct=50):
    """
    Elimina filas o columnas de un DataFrame si el porcentaje de valores nulos supera el umbral especificado.

    Args:
        data_frame: DataFrame de pandas.
        axis: 0 para filas, 1 para columnas.
        max_null_pct: Umbral de porcentaje máximo de valores nulos permitidos.
    """
    # Calcular el número mínimo de valores NO nulos necesarios para no ser eliminado,
    # basado en el porcentaje máximo de valores nulos permitidos.
    if axis == 1:  # Para columnas
        threshold = int((100 - max_null_pct) / 100 * len(data_frame))
    else:  # Para filas
        threshold = int((100 - max_null_pct) / 100 * len(data_frame.columns))
    
    return data_frame.dropna(axis=axis, thresh=threshold)


def impute_nan(data_frame, numeric_strategy='mean', non_numeric_strategy='most_frequent', interpolation_method=None):
    """
    Imputa los valores NaN de un DataFrame utilizando estrategias diferentes para columnas numéricas y no numéricas.

    Args:
        data_frame: DataFrame de pandas.
        numeric_strategy: Estrategia de imputación para columnas numéricas ('mean' o 'median').
        non_numeric_strategy: Estrategia de imputación para columnas no numéricas ('most_frequent' o 'constant').
    """
    print(f'Imputando NaN con estrategias: numérico={numeric_strategy}, no numérico={non_numeric_strategy}')
    data_frame_copy = data_frame.copy()

    numeric_cols = data_frame_copy.select_dtypes(include=['number']).columns

    # Imputación para columnas numéricas
    if numeric_strategy in ['mean', 'median']:
        numeric_imputer = SimpleImputer(strategy=numeric_strategy)
        data_frame_copy[numeric_cols] = numeric_imputer.fit_transform(data_frame_copy[numeric_cols])
    elif numeric_strategy == 'interpolate':
        # Asegura que el método de interpolación se haya proporcionado
        if not interpolation_method:
            raise ValueError("Interpolation method must be specified when numeric_strategy is 'interpolate'.")
        data_frame_copy[numeric_cols] = data_frame_copy[numeric_cols].apply(lambda x: x.interpolate(method=interpolation_method))

    # Imputación para columnas no numéricas utilizando 'most_frequent'
    non_numeric_cols = data_frame_copy.select_dtypes(exclude=['number']).columns
    if non_numeric_strategy == 'most_frequent' and len(non_numeric_cols) > 0:
        non_numeric_imputer = SimpleImputer(strategy='most_frequent')
        data_frame_copy[non_numeric_cols] = non_numeric_imputer.fit_transform(data_frame_copy[non_numeric_cols])

    # Retorna el DataFrame con los valores imputados
    return pd.DataFrame(data_frame_copy, columns=data_frame.columns)


def normalize_data_zscore(data_frame):
    """
    Normaliza los datos de un DataFrame utilizando la puntuación Z (Z-score).
    
    Args:
        data_frame (pd.DataFrame): El DataFrame de pandas a normalizar.
    """
    data_frame_copy = data_frame.copy()
    numeric_cols = data_frame_copy.select_dtypes(include=['number']).columns
    data_frame_copy[numeric_cols] = (data_frame_copy[numeric_cols] - data_frame_copy[numeric_cols].mean()) / data_frame_copy[numeric_cols].std()
    return data_frame_copy


def drop_meaningless_columns(data_frame):
    """
    Elimina columnas innecesarias de un DataFrame: 'id' o 'id_anemo'.

    Args:
        data_frame (pd.DataFrame): El DataFrame de pandas a modificar
    """
    if ("id" in data_frame.columns):
        data_frame.drop(columns=["id"], inplace=True)
    elif ("id_anemo" in data_frame.columns):
        data_frame.drop(columns=["id_anemo"], inplace=True)


def make_datetime_index(data_frame):
    """
    Transforma la columna 'data' de un DataFrame en un índice de tipo datetime.

    Args:
        data_frame (pd.DataFrame): El DataFrame de pandas a modificar
    """
    data_frame['data'] = pd.to_datetime(data_frame['data'], format='%d/%m/%Y %H:%M:%S')
    data_frame.set_index('data', inplace=True)


# TODO: Cambiar la función para que retorne una copia modificada en lugar de modificar el DataFrame original
def modify_dataset_columns(data_frame):
    """
    Realiza modificaciones específicas en las columnas de un DataFrame de pandas. 
    Este proceso incluye dos pasos principales: eliminar columnas innecesarias, como los identificadores, 
    y transformar la columna 'data' en un índice de tipo datetime.
    
    Args:
        data_frame (pd.DataFrame): El DataFrame de pandas a modificar. Esta función 
        asume que `data_frame` es mutable y, por lo tanto, se modifica directamente 
        sin retornar una copia o una versión modificada del mismo.
    """
    drop_meaningless_columns(data_frame)
    make_datetime_index(data_frame)


def save_processed_data(data, name, processed_data_directory_path):
    """
    Guarda un DataFrame de pandas en un archivo CSV en el directorio de datos procesados.

    Args:
        data: DataFrame de pandas.
        name: Nombre base del archivo CSV.
        processed_data_directory_path: Ruta del directorio de datos procesados.
    """
    output_path = processed_data_directory_path / f'{name}.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


def outlier_detection(data_frame, threshold=3):
    """
    Detecta outliers en un DataFrame utilizando el método de puntuación Z (Z-score).

    Args:
        data_frame: DataFrame de pandas.
        threshold: Umbral de puntuación Z para considerar un valor como un outlier.
    """
    # Calcular la puntuación Z de forma diaria para cada columna (La fila son 10 minutos)
    # Agrupar por día y calcular la media y la desviación estándar
    daily_z_scores = data_frame.groupby(data_frame.index.date).transform(lambda x: (x - x.mean()) / x.std())
    # Calcular los outliers basados en la puntuación Z
    return daily_z_scores.abs() > threshold



def outlier_treatment(data_frame, threshold=3, numeric_strategy='interpolate', interpolation_method='linear'):
    """
    Trata los outliers de un DataFrame utilizando el método de puntuación Z (Z-score) para identificarlos
    y luego imputa esos outliers con la estrategia especificada para columnas numéricas.

    Args:
        data_frame: DataFrame de pandas.
        threshold: Umbral de puntuación Z para considerar un valor como outlier.
        numeric_strategy: Estrategia de imputación para columnas numéricas ('mean' o 'median').
    """
    # Identifica outliers
    outliers = outlier_detection(data_frame, threshold=threshold)
    print(f'Número de outliers detectados: {outliers.sum().sum()}')
    
    # Marcar outliers como NaN
    data_frame[outliers] = np.nan

    plot_daily_nulls_per_column(data_frame, "outliers")
    
    # Imputar los NaN (anteriormente outliers) con la estrategia especificada
    return impute_nan(data_frame, numeric_strategy=numeric_strategy, non_numeric_strategy=None, interpolation_method=interpolation_method)



def correlation_matrix(data_frame, name):
    """
    Crea y guarda una matriz de correlación de un DataFrame de pandas.

    Args:
        data_frame: DataFrame de pandas.
        name: Nombre base del archivo.
    """
    correlation_matrix = data_frame.corr()    
    plot_correlation_matrix(correlation_matrix, name)


def process_data(raw_text_file_path):
    """
    Procesa un archivo de texto crudo, realizando las siguientes operaciones:
    - Carga y convierte el archivo a un DataFrame de pandas.
    - Modifica las columnas del DataFrame.
    - Trata los valores NaN.
    - Normaliza los datos utilizando la puntuación Z (Z-score).

    Args:
        raw_text_file_path: Ruta del archivo de texto crudo.
    """
    name, data = load_and_convert_to_csv(raw_text_file_path)
    modify_dataset_columns(data)
    data = treat_nan(data, max_null_column_pct=90, max_null_row_pct=0)
    data = normalize_data_zscore(data)
    data = outlier_treatment(data, threshold=3, numeric_strategy='interpolate', interpolation_method='linear')
    return data, name


def plot_data(data, name):
    """
    Genera y guarda visualizaciones de un conjunto de datos.

    Args:
        data: DataFrame de pandas.
        name: Nombre base del archivo.
    """
    plot_dataset_boxplot(data, name)
    plot_dataset_features(data, name)
    #plot_hourly_feature_per_day(data, name)
    #plot_dataset_boxplot_by_day(data, name)


def main():
    """
    Función principal del script. Procesa los archivos de datos crudos,
    guarda los datos procesados, y genera visualizaciones de los datos.
    """
    raw_data_directory_path = Path('data/raw')
    processed_data_directory_path = Path('data/processed')

    data_names = []
    for file_path in raw_data_directory_path.glob('*.txt'):
        data, name = process_data(file_path)
        correlation_matrix(data, name)
        save_processed_data(data, name, processed_data_directory_path)
        data_names.append((data, name))

    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(plot_data, data, name) for data, name in data_names]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"Se produjo una excepción: {e}")

 

if __name__ == '__main__':
    main()
