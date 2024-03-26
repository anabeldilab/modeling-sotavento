# modeling-sotavento
Study, analysis and modeling of a photovoltaic solar system located in the bioclimatic home of Sotavento Galicia.


## Preprocessing
1. **Drop columns**: Drop columns that are not useful for the models. In this case, 'id' column has been dropped in most of the datasets. Densidad (Torre 12) dataset was the different one, so 'id_anemo' column was dropped instead.

2. **Study of missing data**: Plots were created to study the behavior of missing data in the datasets to choose whether it was convenient to simply discard them or make an imputation. For now, the decision was to discard them. The percentage of missing data was, for some features, 100% and for the rest of the features, less than 1% in all datasets.

3. **Discard missing data**: Missing data was discarded in all datasets. So it was decided to discard them.

4. **Normalization**: The data was normalized using the z-score normalization method. This method is used to normalize the data to have a mean of 0 and a standard deviation of 1. This is done to avoid the problem of different scales in the data.

5. **Study of outliers**: The data was studied to detect outliers. The boxplot method was used to detect them.

4. **Study of the correlation between features**: