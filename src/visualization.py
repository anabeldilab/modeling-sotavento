import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_convert_to_csv(file_path, output_dir):
    base_file_name = Path(file_path).stem
    data_frame = pd.read_csv(file_path, sep=';', engine='python', decimal=',')
    return base_file_name, data_frame


def plot_nan_percentage(data, name):
    print(f" - Processing DataFrame: {name}")
    plt.figure(figsize=(10, 6))
    title = f'NaN Percentage in {name}'
    nan_percentage = data.isnull().sum() / len(data) * 100
    ax = nan_percentage.plot(kind='bar', title=title, ylabel='Percentage of NaN values')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    # Guardar la gráfica en graphs
    output_path = Path(f'graphs/nan_detection/{name}') / f'nan_percentage_{name}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_hourly_nulls_per_day(data_frame, date_column, title):
    data_frame = data_frame.copy()
    
    # Identificar columnas que no son completamente nulas y tienen al menos un valor nulo
    cols_with_some_nulls = [col for col in data_frame.columns if data_frame[col].isnull().any() and not data_frame[col].isnull().all()]


    for col in cols_with_some_nulls:
        print(f"Processing column: {col}")

        # Contar nulos de la col en cuestión
        data_frame['partial_nulls'] = data_frame[col].isnull()
        # Me quedo con los true
        partial_nulls = data_frame[data_frame['partial_nulls']].copy()
        # Obtener el día y la hora de cada nulo parcial
        partial_nulls['date'] = partial_nulls[date_column].dt.date
        partial_nulls['hour'] = partial_nulls[date_column].dt.hour

        # Obtener las fechas únicas para iterar sobre ellas
        unique_dates = partial_nulls['date'].unique()

        for date in unique_dates:
            # Filtrar por fecha específica
            daily_data = partial_nulls[partial_nulls['date'] == date]
            # Contar la cantidad de nulos parciales por hora
            daily_data = daily_data.groupby('hour')['partial_nulls'].count().reset_index()

            # Graficar
            plt.figure(figsize=(12, 8))
            plt.bar(daily_data['hour'], daily_data['partial_nulls'], color='skyblue', width=1.0)
            plt.xticks(range(0, 24), [f"{hour}:00" for hour in range(0, 24)], rotation=45)
            plt.title(f'Presencia de Nulos Parciales en {col} por Hora del Día - {date} - {title}')
            plt.xlabel('Hora del Día')
            plt.ylabel('Cantidad de Horas con Algunos Nulos')
            plt.grid(axis='y', linestyle='--')
            plt.tight_layout()

            # Guardar la gráfica
            output_path = Path(f'graphs/nan_detection/{title}/{col}') / f'hourly_partial_nulls_{date}_{title}.png'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)
            plt.close()


def plot_daily_nulls_per_column(data_frame, date_column, title):
    data_frame = data_frame.copy()
    
    # Crear un DataFrame con cada día del año en el rango del dataset
    start_date = data_frame[date_column].dt.date.min()
    end_date = data_frame[date_column].dt.date.max()
    all_dates = pd.DataFrame({'date': pd.date_range(start=start_date, end=end_date, freq='D')})
    
    # Convertir 'date' en all_dates a tipo date para compatibilidad
    all_dates['date'] = all_dates['date'].dt.date
    
    # Identificar columnas que no son completamente nulas y tienen al menos un valor nulo
    cols_with_some_nulls = [col for col in data_frame.columns if data_frame[col].isnull().any() and not data_frame[col].isnull().all()]

    for col in cols_with_some_nulls:
        # Contar los nulos en la columna actual por día
        data_frame['is_null'] = data_frame[col].isnull().astype(int)
        nulls_by_day = data_frame.groupby(data_frame[date_column].dt.date)['is_null'].sum().reset_index()
        nulls_by_day.rename(columns={'is_null': 'count', date_column: 'date'}, inplace=True)

        # Fusionar con el DataFrame de todos los días, asegurando todos los días del año estén presentes
        daily_nulls = pd.merge(all_dates, nulls_by_day, on='date', how='left').fillna(0)

        # Graficar
        plt.figure(figsize=(15, 6))
        plt.plot(daily_nulls['date'], daily_nulls['count'], marker='o', linestyle='-', color='skyblue', markersize=2)
        plt.title(f'Nulos Diarios en "{col}" - {title}')
        plt.xlabel('Fecha')
        plt.ylabel('Cantidad de Nulos')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()

        # Guardar la gráfica
        base_output_path = Path(f'graphs/nan_detection/{title}')
        base_output_path.mkdir(parents=True, exist_ok=True)
        output_path = base_output_path / f'daily_nulls_{col}_{title}.png'
        plt.savefig(output_path)
        plt.close()


def plot_boxplot(x, data, boxplot_name="boxplot", title="Data"):  # x = ['Modelo 1', 'Modelo 2', 'Modelo 3', 'Modelo 4', 'Modelo 5'] # data = [score1, score2, score3, score4, score5]
    """
    Crea un boxplot con los data de los modelos
    
    Args:
    x (list): Lista con los nombres de los modelos
    data (list): Lista con los data de los modelos
    boxplot_name (str): Nombre del archivo de salida    
    """
    fig7, ax = plt.subplots()
    ax.set_title(f'Boxplot de {boxplot_name} - {title}')
    ax.boxplot(data, labels=x)
    ax.set_xticklabels(x, rotation=45, ha="right")
    ax.set_ylabel('Puntuación')
    plt.tight_layout()
    base_output_path = Path(f'graphs/outliers_detection/{title}/boxplots')
    base_output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{base_output_path}/boxplot-{boxplot_name}.png')
    plt.close()


# CSV con los nan de cada feature
def create_nan_csv(data_frame, name):
    nan_df = data_frame.isnull().sum()
    nan_df = nan_df[nan_df > 0]
    total = len(data_frame)
    nan_df = nan_df / total * 100
    nan_df = nan_df.to_frame().T
    base_output_path = Path(f'data/nan_detection')
    base_output_path.mkdir(parents=True, exist_ok=True)
    nan_df.to_csv(f'{base_output_path}/{name}_nan.csv')


def plot_dataset_boxplot(data_frame, title):
    print(f" - Processing DataFrame: {title}")
    columns = data_frame.columns.to_list()
    print(f"Columns: {columns}")
    if 'data' in columns:
        columns.remove('data')
    
    for feature_name in columns:
        feature_data = data_frame[feature_name].values
        plot_boxplot([feature_name], feature_data, feature_name, title)


def plot_dataset_features(data_frame, title):
    # Make a graph for each feature
    for feature in data_frame.columns:
        if feature == 'data':
            continue
        plt.figure(figsize=(12, 8))
        plt.plot(data_frame[feature], marker='o', linestyle='', color='skyblue', markersize=2)
        plt.title(f'{feature} - {title}')
        plt.xlabel('Fecha')
        plt.ylabel('Valor')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        base_output_path = Path(f'graphs/outliers_detection/{title}/year')
        base_output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{base_output_path}/graph-{feature}.png')
        plt.close()


def plot_hourly_feature_per_day(data_frame, date_column, title):
    data_frame = data_frame.copy()
    
    for feature_column in data_frame.columns:
        # Asegurarse de que la columna de interés no sea completamente nula
        if data_frame[feature_column].isnull().all():
            print(f"Column {feature_column} is completely null.")
            return
        
        # Añadir columnas para fecha y hora
        data_frame['date'] = data_frame.index.date
        data_frame['hour'] = data_frame.index.hour
            
        # Obtener las fechas únicas para iterar sobre ellas
        unique_dates = data_frame['date'].unique()
        
        for date in unique_dates:
            # Filtrar por fecha específica
            daily_data = data_frame[data_frame['date'] == date]
            # Contar la cantidad de registros por hora (podrías cambiar esto por sum(), mean(), etc.)
            daily_data = daily_data.groupby('hour')[feature_column].mean().reset_index()

            # Graficar
            plt.figure(figsize=(12, 8))
            plt.scatter(daily_data['hour'], daily_data[feature_column], color='skyblue')
            plt.xticks(range(0, 24), [f"{hour}:00" for hour in range(0, 24)], rotation=45)
            plt.title(f'{title} - {feature_column} por Hora del Día - {date}')
            plt.xlabel('Hora del Día')
            plt.ylabel(f'Cantidad de {feature_column}')
            plt.grid(axis='y', linestyle='--')
            plt.tight_layout()

            # Guardar la gráfica
            year_month = pd.to_datetime(date).strftime('%Y_%m')
            output_path = Path(f'graphs/outliers_detection/{title}/daily/{year_month}/{feature_column}') / f'hourly_feature_scatter_{date}_{title}.png'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)
            plt.close()

def plot_correlation_matrix(correlation_matrix, title):
    # Configuración para mejorar la visualización
    plt.figure(figsize=(10, 8))

    # Crear el mapa de calor
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.05)

    # Títulos y etiquetas
    plt.title(f'Matriz de Correlación de {title}')
    output_path = Path(f'graphs/correlation_matrix') / f'correlation_matrix_{title}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
