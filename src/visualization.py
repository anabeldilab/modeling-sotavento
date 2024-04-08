import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(data, plot_type='bar', title='', xlabel='', ylabel='', save_path=None, **kwargs):
    """
    Crea y guarda un gráfico basado en los parámetros proporcionados.
    
    Args:
        data: Datos a graficar. La forma esperada depende del tipo de gráfico.
        plot_type (str): Tipo de gráfico ('bar', 'line', 'scatter', 'box', 'heatmap').
        title (str): Título del gráfico.
        xlabel (str): Etiqueta para el eje X.
        ylabel (str): Etiqueta para el eje Y.
        save_path (Path or str, optional): Ruta donde guardar el gráfico. Si es None, no se guarda.
        **kwargs: Argumentos clave-valor adicionales específicos de cada tipo de gráfico.
    """
    plt.figure(figsize=kwargs.pop('figsize', (10, 6)))
    xticks = kwargs.pop('xticks', None)
    rotation = kwargs.pop('rotation', None)
    xticklabels = kwargs.pop('xticklabels', None)
    
    if plot_type == 'bar':
        sns.barplot(data=data, **kwargs)
    elif plot_type == 'line':
        sns.lineplot(data=data, **kwargs)
    elif plot_type == 'scatter':
        sns.scatterplot(data=data, **kwargs)
    elif plot_type == 'box':
        sns.boxplot(data=data, **kwargs)
    elif plot_type == 'heatmap':
        sns.heatmap(data=data, **kwargs)
    else:
        raise ValueError(f"Unsupported plot_type: {plot_type}")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xticks is not None and xticklabels is not None:
        plt.xticks(ticks=xticks, labels=xticklabels, rotation=rotation)

    plt.tight_layout()
    
    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def plot_nan_percentage(data, name):
    print(f" - Processing DataFrame: {name}")
    nan_percentage_per_feature = data.isnull().sum() / len(data) * 100
    plot_data(
        data=nan_percentage_per_feature,
        plot_type='bar',
        title=f'NaN Percentage in {name}',
        xlabel='Columns',
        ylabel='Percentage of NaN values',
        save_path=f'graphs/nan_detection/{name}/nan_percentage_{name}.png'
    )


def plot_hourly_nulls_per_day(data_frame, title):
    data_frame = data_frame.copy()
    
    # Identificar columnas que no son completamente nulas y tienen al menos un valor nulo
    cols_with_some_nulls = [col for col in data_frame.columns if data_frame[col].isnull().any() and not data_frame[col].isnull().all()]

    for col in cols_with_some_nulls:
        print(f"Processing column: {col}")

        # Contar nulos de la columna en cuestión
        data_frame['partial_nulls'] = data_frame[col].isnull()
        # Filtrar para quedarse solo con los registros con nulos
        partial_nulls = data_frame[data_frame['partial_nulls']].copy()
        # Obtener el día y la hora de cada nulo parcial
        partial_nulls['date'] = partial_nulls.index.date
        partial_nulls['hour'] = partial_nulls.index.hour

        # Obtener las fechas únicas para iterar sobre ellas
        unique_dates = partial_nulls['date'].unique()

        for date in unique_dates:
            # Filtrar por fecha específica
            daily_data = partial_nulls[partial_nulls['date'] == date]
            # Contar la cantidad de nulos parciales por hora
            daily_data = daily_data.groupby('hour')['partial_nulls'].count().reset_index(name='count')

            plot_data(
                data=daily_data, 
                plot_type='bar',
                title=f'Presencia de Nulos Parciales en {col} por Hora del Día - {date} - {title}',
                xlabel='Hora del Día',
                ylabel='Cantidad de Horas con Algunos Nulos',
                save_path=f'graphs/nan_detection/{title}/{col}/hourly_partial_nulls_{date}_{title}.png',
                x='hour',  # Argumento de seaborn para el eje X
                y='count',  # Argumento de seaborn para el eje Y
                color='skyblue',  # Argumento de seaborn para el color
                figsize=(12, 8)  # Argumento de la función plot_data para el tamaño de figura
            )


def plot_daily_nulls_per_column(data_frame, title):
    data_frame = data_frame.copy()

    # Identificar las columnas que tienen al menos un valor nulo pero no son completamente nulas
    cols_with_some_nulls = [col for col in data_frame.columns if data_frame[col].isnull().any() and not data_frame[col].isnull().all()]

    # Crear un DataFrame con cada día en el rango del índice del DataFrame original
    start_date = data_frame.index.min()
    end_date = data_frame.index.max()
    all_dates = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='D'))
    all_dates['date'] = all_dates.index.date  # Añadir una columna de fechas como date objects para facilitar la fusión

    for col in cols_with_some_nulls:
        # Contar los nulos en la columna actual por día
        data_frame['is_null'] = data_frame[col].isnull().astype(int)

        nulls_by_day = data_frame.groupby(data_frame.index.date)['is_null'].sum().reset_index(name='count')
        nulls_by_day.rename(columns={'index': 'date'}, inplace=True)

        # Fusionar con el DataFrame de todos los días, asegurando todos los días del año estén presentes
        daily_nulls = pd.merge(all_dates, nulls_by_day, on='date', how='left').fillna(0)

        plot_data(
            data=daily_nulls,
            plot_type='line',
            title=f'Nulos Diarios en "{col}" - {title}',
            xlabel='Fecha',
            ylabel='Cantidad de Nulos',
            save_path=f'graphs/nan_detection/{title}/daily_nulls_{col}_{title}.png',
            figsize=(15, 6)
        )
        

def plot_dataset_boxplot(data_frame, title):
    """
    Genera y guarda boxplots para cada característica numérica de un DataFrame.

    Esta función itera a través de cada columna del DataFrame proporcionado y crea un boxplot para los valores de
    cada característica.

    Args:
        data_frame (pd.DataFrame): DataFrame que contiene las características (columnas)
            para las cuales se generarán los boxplots. Puede contener cualquier número de
            columnas numéricas y una columna 'data' que, si existe, será excluida de la
            visualización.
        title (str): Título base para los gráficos generados. Este título se complementa con
            el nombre de la característica para cada gráfico generado, proporcionando un
            contexto claro sobre qué característica está siendo visualizada.

    Ejemplo de Uso:
        plot_dataset_boxplot(data_frame=df, title='Análisis de Features')
    """
    print(f" - Processing DataFrame: {title}")
    columns = data_frame.columns.to_list()
    print(f"Columns: {columns}")    
    for feature_name in columns:
        feature_data = data_frame[[feature_name]].melt(var_name='Feature', value_name='Value')
        
        plot_data(
            data=feature_data,
            plot_type='box',
            title=f'Boxplot of {feature_name} - {title}',
            xlabel='Feature',
            ylabel='Value',
            save_path=f'graphs/outliers_detection/{title}/boxplots/boxplot-{feature_name}.png',
            x='Feature',
            y='Value'
        )
        

def plot_dataset_boxplot_by_day(data_frame, title):
    """
    Genera y guarda boxplots diarios para cada característica (feature) del DataFrame,
    asumiendo que el índice del DataFrame es de tipo datetime y representa las fechas.
    
    Esta función crea un boxplot para cada día presente en el índice del DataFrame y para
    cada característica numérica, permitiendo la visualización de la distribución diaria
    de los valores de cada característica. 

    Args:
        data_frame (pd.DataFrame): DataFrame que contiene los datos a ser analizados.
            El índice del DataFrame debe ser de tipo datetime, ya que se utiliza para
            agrupar los datos por día. Las columnas deben representar características
            numéricas que se desean analizar.
        title (str): Título base para los gráficos generados. Este título se complementa
            con el nombre de la característica y la fecha para cada gráfico generado,
            facilitando la identificación y comparación de distribuciones diarias.

    Notas:
        - Es importante que el índice del DataFrame esté en formato datetime para
          que la agrupación por día funcione correctamente.
        - Los gráficos generados se guardan en una estructura de directorios organizada
          por el título del análisis, año y mes, y finalmente por característica, lo que
          ayuda a mantener los resultados organizados y accesibles para futuras consultas.
    
    Ejemplo de Uso:
        plot_dataset_boxplot_by_day(data_frame=df, title='Análisis Preliminar Diario')
    """
    print(f" - Processing DataFrame: {title} - Boxplot por Día")
    data_frame = data_frame.copy()

    for feature_column in data_frame.columns:
        print(f"Processing feature: {feature_column}")
        # Excluir cualquier columna no numérica de las características a graficar.
        if pd.api.types.is_numeric_dtype(data_frame[feature_column]):
            # Agrupar los datos por fecha (índice) y preparar los datos para cada día.
            grouped_data = data_frame.groupby(data_frame.index.date)[feature_column].apply(list).reset_index(name='values')
            grouped_data['date'] = pd.to_datetime(grouped_data['index'])

            for _, row in grouped_data.iterrows():
                date = row['index']
                daily_data = pd.DataFrame({feature_column: row['values']})
                
                plot_data(
                    data=daily_data,
                    plot_type='box',
                    title=f'{title} - {feature_column} por Día - {date.strftime("%Y-%m-%d")}',
                    xlabel='Día',
                    ylabel=f'Valores de {feature_column}',
                    save_path=f'graphs/outliers_detection/{title}/boxplot/{feature_column}/daily/{date.strftime("%Y_%m")}/boxplot_{date.strftime("%Y_%m_%d")}_{title}.png',
                    figsize=(10, 6)
                )


def plot_dataset_features(data_frame, title):
    """
    Genera y guarda un gráfico para cada característica del DataFrame.

    Args:
        data_frame (pd.DataFrame): DataFrame con los datos a graficar.
        title (str): Título para usar en los gráficos y nombres de archivos.

    Ejemplo de Uso:
        plot_dataset_features(data_frame=df, title='Análisis de Features')
    """
    for feature in data_frame.columns:
        print("Processing feature:", feature)
        # Construir la ruta de guardado del gráfico
        base_output_path = Path(f'graphs/outliers_detection/{title}/scatter/year')
        save_path = base_output_path / f'graph-{feature}.png'

        plot_data(
            data=data_frame,
            plot_type='scatter',
            title=f'{feature} - {title}',
            xlabel='Fecha',
            ylabel='Valor',
            save_path=save_path,
            figsize=(12, 8),
            edgecolor="none",
            x='data', 
            y=feature,
            color='skyblue',  # Color de los puntos o línea
            marker='o',  # Tipo de marcador, solo aplica para scatter
            linestyle=''
        )


def plot_hourly_feature_per_day(data_frame, title):
    print(f" - Processing DataFrame: {title} - Hourly Feature per Day")
    data_frame = data_frame.copy()
    
    # Obtener las fechas únicas para iterar sobre ellas
    unique_dates = pd.Series(data_frame.index.date).unique()

    for feature_column in data_frame.columns:
        print(f"Processing feature: {feature_column}")
        # Asegurarse de que la columna de interés no sea completamente nula
        if data_frame[feature_column].isnull().all():
            print(f"Column {feature_column} is completely null.")
            return
        
        for date in unique_dates:
            # Filtrar por fecha específica
            daily_data = data_frame[data_frame.index.date == date]
            # Contar la cantidad de registros por hora (podrías cambiar esto por sum(), mean(), etc.)
            hourly_data = daily_data.groupby(daily_data.index.hour)[feature_column].mean().reset_index()

            x_ticks = list(range(0, 24))  # Horas del día.
            x_tick_labels = [f"{hour}:00" for hour in x_ticks]  # Etiquetas personalizadas para las horas.

            plot_data(
                data=hourly_data,
                plot_type='scatter',
                title=f'{title} - {feature_column} por Hora del Día - {date}',
                xlabel='Hora del Día',
                ylabel=f'Promedio de {feature_column}',
                save_path=f'graphs/outliers_detection/{title}/scatter/{feature_column}/daily/{date.strftime("%Y_%m")}/hourly_feature_scatter_{date}_{title}.png',
                x=hourly_data.index,
                y=feature_column,
                edgecolor="none",
                xticks=x_ticks,
                rotation=45,
                xticklabels=x_tick_labels,
                color='skyblue'
            )


def plot_correlation_matrix(correlation_matrix, title):
    plot_data(
        data=correlation_matrix,
        plot_type='heatmap',
        title=f'Matriz de Correlación de {title}',
        save_path=f'graphs/correlation_matrix/correlation_matrix_{title}.png',
        annot=True,  # Muestra los valores dentro de cada celda
        cmap='coolwarm',  # Colormap
        fmt=".2f",  # Formato de los números dentro de las celdas
        linewidths=.05  # Ancho de las líneas que separan las celdas
    )
