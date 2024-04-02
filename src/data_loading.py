import pandas as pd
from pathlib import Path

def load_and_convert_to_csv(file_path, sep=';', decimal=','):
    """
    Carga un archivo CSV, usando parámetros personalizados para el separador y el decimal,
    y devuelve un DataFrame junto con el nombre base del archivo.

    Args:
        file_path: Ruta del archivo CSV.
        sep: Separador de columnas.
        decimal: Caracter decimal.
    """
    base_file_name = Path(file_path).stem
    data_frame = pd.read_csv(file_path, sep=sep, engine='python', decimal=decimal)
    return base_file_name, data_frame


# CSV con los nan de cada feature
def create_nan_csv(data_frame, name):
    """
    Crea un archivo CSV con la cantidad de valores NaN por columna de un DataFrame.

    Args:
        data_frame: DataFrame de pandas.
        name: Nombre base del archivo CSV.
    """
    nan_df = data_frame.isnull().sum()
    nan_df = nan_df[nan_df > 0]
    total = len(data_frame)
    nan_df = nan_df / total * 100
    nan_df = nan_df.to_frame().T
    base_output_path = Path(f'data/nan_detection')
    base_output_path.mkdir(parents=True, exist_ok=True)
    nan_df.to_csv(f'{base_output_path}/{name}_nan.csv')
