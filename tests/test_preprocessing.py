import unittest
from unittest.mock import patch, mock_open
from pathlib import Path
from src.preprocessing import load_and_convert_to_csv, plot_nan_percentage
from unittest.mock import patch, MagicMock

class TestPreprocessing(unittest.TestCase):
    @patch("pandas.read_csv")
    @patch("pandas.DataFrame.to_csv")
    def test_load_and_convert_to_csv(self, mock_to_csv, mock_read_csv):
        # Configura el mock para read_csv para que retorne un DataFrame de pandas simulado
        mock_df = mock_read_csv.return_value
        
        # Simula la función open para no leer realmente un archivo
        with patch("builtins.open", mock_open()):
            load_and_convert_to_csv("dummy_path/raw_file.txt", Path("dummy_path/processed"))

        # Verifica que read_csv fue llamado con el path correcto
        mock_read_csv.assert_called_once_with("dummy_path/raw_file.txt", sep=';', engine='python', decimal=',')

        # Verifica que to_csv fue llamado una vez (indicando que el archivo se intentó guardar)
        mock_df.to_csv.assert_called_once()


'''    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_plot_nan_percentage(self, mock_figure, mock_show):
        # Prepara un objeto MagicMock para simular el resultado de data.isnull().sum() / len(data) * 100
        mock_nan_percentage_series = MagicMock(name='Mock Series for NaN Percentage')
        
        # Configura el DataFrame mock para simular la cadena de llamadas .isnull().sum() y luego simular el resultado de la división
        mock_df = MagicMock()
        mock_df.isnull.return_value.sum.return_value = mock_nan_percentage_series

        # Llama a la función bajo prueba
        plot_nan_percentage(mock_df, "dummy_name")

        # Verifica si el método plot se llamó en el objeto mockeado
        mock_nan_percentage_series.plot.assert_called_once_with(kind='bar', title=f'NaN Percentage in dummy_name', ylabel='Percentage of NaN values')
''' 


if __name__ == '__main__':
    unittest.main()
