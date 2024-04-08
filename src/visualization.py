import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(data, plot_type='bar', title='', xlabel='', ylabel='', save_path=None, **kwargs):
    """
    Creates and saves a plot based on the provided parameters.
    
    Args:
        data: Data to plot. The expected shape depends on the plot type.
        plot_type (str): Type of plot ('bar', 'line', 'scatter', 'box', 'heatmap').
        title (str): Plot title.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        save_path (Path or str, optional): Path to save the plot. If None, the plot is not saved.
        **kwargs: Additional key-value arguments specific to each plot type.
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
    
    # Identify columns that are not completely null and have at least one null value
    cols_with_some_nulls = [col for col in data_frame.columns if data_frame[col].isnull().any() and not data_frame[col].isnull().all()]

    for col in cols_with_some_nulls:
        print(f"Processing column: {col}")

        # Count nulls for the concerned column
        data_frame['partial_nulls'] = data_frame[col].isnull()
        # Filter to only keep records with nulls
        partial_nulls = data_frame[data_frame['partial_nulls']].copy()
        # Get the day and hour of each partial null
        partial_nulls['date'] = partial_nulls.index.date
        partial_nulls['hour'] = partial_nulls.index.hour

        # Get the unique dates to iterate over them
        unique_dates = partial_nulls['date'].unique()

        for date in unique_dates:
            # Filter by specific date
            daily_data = partial_nulls[partial_nulls['date'] == date]
            # Count the number of partial nulls per hour
            daily_data = daily_data.groupby('hour')['partial_nulls'].count().reset_index(name='count')

            plot_data(
                data=daily_data, 
                plot_type='bar',
                title=f'Presence of Partial Nulls in {col} by Hour of the Day - {date} - {title}',
                xlabel='Hour of the Day',
                ylabel='Number of Hours with Some Nulls',
                save_path=f'graphs/nan_detection/{title}/{col}/hourly_partial_nulls_{date}_{title}.png',
                x='hour',  # Seaborn argument for X-axis
                y='count',  # Seaborn argument for Y-axis
                color='skyblue',  # Seaborn argument for color
                figsize=(12, 8)  # plot_data function argument for figure size
            )


def plot_daily_nulls_per_column(data_frame, title):
    data_frame = data_frame.copy()

    # Identify columns that have at least one null value but are not completely null
    cols_with_some_nulls = [col for col in data_frame.columns if data_frame[col].isnull().any() and not data_frame[col].isnull().all()]

    # Create a DataFrame with each day in the range of the original DataFrame's index
    start_date = data_frame.index.min()
    end_date = data_frame.index.max()
    all_dates = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='D'))
    all_dates['date'] = all_dates.index.date  # Add a column of dates as date objects for easier merging

    for col in cols_with_some_nulls:
        # Count the nulls in the current column by day
        data_frame['is_null'] = data_frame[col].isnull().astype(int)

        nulls_by_day = data_frame.groupby(data_frame.index.date)['is_null'].sum().reset_index(name='count')
        nulls_by_day.rename(columns={'index': 'date'}, inplace=True)

        # Merge with the DataFrame of all days, ensuring every day of the year is present
        daily_nulls = pd.merge(all_dates, nulls_by_day, on='date', how='left').fillna(0)

        plot_data(
            data=daily_nulls,
            plot_type='line',
            title=f'Daily Nulls in "{col}" - {title}',
            xlabel='Date',
            ylabel='Number of Nulls',
            save_path=f'graphs/nan_detection/{title}/daily_nulls_{col}_{title}.png',
            figsize=(15, 6)
        )
        

def plot_dataset_boxplot(data_frame, title):
    """
    Generates and saves boxplots for each numeric feature of a DataFrame.

    This function iterates through each column of the provided DataFrame and creates a boxplot for the values of
    each feature.

    Args:
        data_frame (pd.DataFrame): DataFrame containing the features (columns)
            for which the boxplots will be generated. It may contain any number of
            numeric columns and a 'data' column which, if present, will be excluded from the
            visualization.
        title (str): Base title for the generated plots. This title is complemented with
            the feature name for each generated plot, providing clear context on which feature is being visualized.

    Usage Example:
        plot_dataset_boxplot(data_frame=df, title='Feature Analysis')
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
    Generates and saves daily boxplots for each feature of the DataFrame,
    assuming the DataFrame's index is datetime type and represents dates.
    
    This function creates a boxplot for each day present in the DataFrame's index and for
    each numeric feature, allowing the visualization of the daily distribution of each feature's values. 

    Args:
        data_frame (pd.DataFrame): DataFrame containing the data to be analyzed.
            The DataFrame's index must be datetime type, as it is used to
            group the data by day. The columns should represent numeric features to be analyzed.
        title (str): Base title for the generated plots. This title is complemented
            with the feature name and the date for each generated plot, facilitating the identification and comparison of daily distributions.

    Notes:
        - It's important that the DataFrame's index is in datetime format for
          the grouping by day to work correctly.
        - The generated plots are saved in a directory structure organized
          by the analysis title, year and month, and finally by feature, which
          helps keep the results organized and accessible for future reference.
    
    Usage Example:
        plot_dataset_boxplot_by_day(data_frame=df, title='Preliminary Daily Analysis')
    """
    print(f" - Processing DataFrame: {title} - Boxplot by Day")
    data_frame = data_frame.copy()

    for feature_column in data_frame.columns:
        print(f"Processing feature: {feature_column}")
        # Exclude any non-numeric column from the features to plot.
        if pd.api.types.is_numeric_dtype(data_frame[feature_column]):
            # Group data by date (index) and prepare the data for each day.
            grouped_data = data_frame.groupby(data_frame.index.date)[feature_column].apply(list).reset_index(name='values')
            grouped_data['date'] = pd.to_datetime(grouped_data['index'])

            for _, row in grouped_data.iterrows():
                date = row['index']
                daily_data = pd.DataFrame({feature_column: row['values']})
                
                plot_data(
                    data=daily_data,
                    plot_type='box',
                    title=f'{title} - {feature_column} per Day - {date.strftime("%Y-%m-%d")}',
                    xlabel='Day',
                    ylabel=f'Values of {feature_column}',
                    save_path=f'graphs/outliers_detection/{title}/boxplot/{feature_column}/daily/{date.strftime("%Y_%m")}/boxplot_{date.strftime("%Y_%m_%d")}_{title}.png',
                    figsize=(10, 6)
                )


def plot_dataset_features(data_frame, title):
    """
    Generates and saves a plot for each feature of the DataFrame.

    Args:
        data_frame (pd.DataFrame): DataFrame with the data to plot.
        title (str): Title to use in the plots and file names.

    Usage Example:
        plot_dataset_features(data_frame=df, title='Feature Analysis')
    """
    for feature in data_frame.columns:
        print("Processing feature:", feature)
        # Build the graph save path
        base_output_path = Path(f'graphs/outliers_detection/{title}/scatter/year')
        save_path = base_output_path / f'graph-{feature}.png'

        plot_data(
            data=data_frame,
            plot_type='scatter',
            title=f'{feature} - {title}',
            xlabel='Date',
            ylabel='Value',
            save_path=save_path,
            figsize=(12, 8),
            edgecolor="none",
            x='data', 
            y=feature,
            color='skyblue',  # Color of points or line
            marker='o',  # Marker type, only applies to scatter
            linestyle=''
        )


def plot_hourly_feature_per_day(data_frame, title):
    print(f" - Processing DataFrame: {title} - Hourly Feature per Day")
    data_frame = data_frame.copy()
    
    # Get the unique dates to iterate over them
    unique_dates = pd.Series(data_frame.index.date).unique()

    for feature_column in data_frame.columns:
        print(f"Processing feature: {feature_column}")
        # Ensure that the column of interest is not completely null
        if data_frame[feature_column].isnull().all():
            print(f"Column {feature_column} is completely null.")
            return
        
        for date in unique_dates:
            # Filter by specific date
            daily_data = data_frame[data_frame.index.date == date]
            # Count the number of records per hour (you could change this to sum(), mean(), etc.)
            hourly_data = daily_data.groupby(daily_data.index.hour)[feature_column].mean().reset_index()

            x_ticks = list(range(0, 24))  # Hours of the day.
            x_tick_labels = [f"{hour}:00" for hour in x_ticks]  # Custom labels for the hours.

            plot_data(
                data=hourly_data,
                plot_type='scatter',
                title=f'{title} - {feature_column} per Hour of the Day - {date}',
                xlabel='Hour of the Day',
                ylabel=f'Average of {feature_column}',
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
        title=f'Correlation Matrix of {title}',
        save_path=f'graphs/correlation_matrix/correlation_matrix_{title}.png',
        annot=True,  # Show values inside each cell
        cmap='coolwarm',  # Colormap
        fmt=".2f",  # Number format inside the cells
        linewidths=.05  # Width of the lines that separate the cells
    )
