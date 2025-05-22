import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from service.technical_indicators_service import TechnicalIndicatorsService
from service.display_results_service import DisplayResultsService
import parameters






class PrepareDatasetService:
    """ Processing on the dataset before model training """



    def __init__(self):
        pass
        parameters.DATASET_PATH



    def format_dataset(self, initial_dataset):
        """ Data preparation """
        tmp_dataset = initial_dataset.copy()
        tmp_dataset['Date'] = pd.to_datetime(initial_dataset['Date'], format='%d/%m/%Y', errors='coerce')
        tmp_dataset = tmp_dataset.sort_values(by='Date')
        numeric_columns = ["Dernier", "Ouv.", " Plus Haut", "Plus Bas", "Variation %"]
        for col in numeric_columns:
            tmp_dataset[col] = tmp_dataset[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        for col in numeric_columns:
            tmp_dataset[col] = pd.to_numeric(tmp_dataset[col], errors='coerce')
        return tmp_dataset



    def delete_columns(self, tmp_dataset):
        """ Removal of columns from the original dataset """
        tmp_dataset = tmp_dataset.drop(columns=['Vol.', 'Variation %', 'Ouv.', ' Plus Haut', 'Plus Bas'])
        return tmp_dataset



    def add_technicals_indicators(self, tmp_dataset):
        """ Adding technical indicators to the dataset (without adding SMA crossings) """
        # Adding indicators :
        tmp_dataset['MA_150'] = TechnicalIndicatorsService.ma(tmp_dataset, 150)
        tmp_dataset['MA_100'] = TechnicalIndicatorsService.ma(tmp_dataset, 100)
        tmp_dataset['MA_50'] = TechnicalIndicatorsService.ma(tmp_dataset, 50)
        tmp_dataset['RSI'] = TechnicalIndicatorsService.rsi(tmp_dataset, 14)
        # Remove rows where MA_150 is NaN:
        tmp_dataset = tmp_dataset.dropna(subset=['MA_150'])
        return tmp_dataset



    def get_fitted_scaler(self, tmp_dataset):
        """ Adjusts the scaler """
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(tmp_dataset)
        return scaler



    def normalize_datas(self, tmp_dataset, scaler):
        """ Normalizes the dataset """
        return scaler.transform(tmp_dataset)



    def create_train_and_test_dataset(self, model_dataset):
        """ Creation of training and test datasets """
        training_size = int(len(model_dataset) * 0.60)
        train_data, test_data = model_dataset.iloc[0:training_size, :], model_dataset.iloc[training_size:len(model_dataset), :]
        return train_data, test_data



    def create_dataset(self, dataset, time_step=1):
        """ Generates multi-dimensional training and test datasets """
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset.iloc[i:(i + time_step)].values 
            dataX.append(a)
            dataY.append(dataset.iloc[i + time_step, 0])
        return np.array(dataX), np.array(dataY)



    def create_one_dimension_dataset(self, dataset, time_step=1):
        """ Generates training and test datasets at one dimension """
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset.iloc[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset.iloc[i + time_step, 0])
        return np.array(dataX), np.array(dataY)



    def create_data_matrix(self, dataset, time_step=15):
        """ Creation of data matrices """
        # Creating datasets :
        x, y = self.create_dataset_for_matrix(dataset, time_step)
        x = x.reshape(x.shape[0], x.shape[1], 1)
        # Displaying dimensions of the datasets:
        print("dataset x: ", x.shape, "dataset y: ", y.shape)
        return x, y



    def create_dataset_for_matrix(self, dataset, time_step=1):
        """ Generates training and test datasets for a matrix """
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset.iloc[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset.iloc[i + time_step, 0])
        return np.array(dataX), np.array(dataY)



    def subsample_old_data(self, tmp_dataset, cutoff_date, fraction=0.1):
        """ Subsampling of old data """
        old_data = tmp_dataset[tmp_dataset['Date'] < cutoff_date]
        recent_data = tmp_dataset[tmp_dataset['Date'] >= cutoff_date]
        old_data_sampled = old_data.sample(frac=fraction, random_state=42)
        combined_data = pd.concat([old_data_sampled, recent_data])
        combined_data = combined_data.sort_values(by='Date').reset_index(drop=True)
        return combined_data



    def add_lag_features(self, dataset, lags):
        """ Adding lags """
        for lag in lags:
            dataset[f'Lag_{lag}'] = dataset['Dernier'].shift(lag)
        return dataset



    def calculate_historical_volatility(self, dataset, window=252):
        """ Calculation of historical volatility """
        # Calculation of logarithmic returns :
        dataset['Returns'] = np.log(dataset['Dernier'] / dataset['Dernier'].shift(1))
        # Calculation of historical volatility :
        dataset['Historical_Volatility'] = dataset['Returns'].rolling(window=window).std() * np.sqrt(252)
        # Removal of the temporary 'Returns' column :
        dataset.drop(columns=['Returns'], inplace=True)
        return dataset



    def save_tmp_dataset(self, dataset):
        """ Saving the dataset to a .csv file """
        saved_dataset = pd.DataFrame(dataset)
        saved_dataset.to_csv(parameters.FORMATED_BTC_COTATIONS_FILE, index=False, encoding='utf-8')



    def prepare_many_dimensions_dataset(self, dataset, cutoff_date='2020-01-01', lags=None, add_volatility=False):
        """ Preparation of the multi-dimensional dataset before training """
        # Preparation of the dataset :
        tmp_dataset = self.format_dataset(dataset)
        tmp_dataset = self.delete_columns(tmp_dataset)
        # Saving the formatted dataset :
        self.save_tmp_dataset(tmp_dataset)
        # Adding technical indicators to the dataset :
        tmp_dataset = self.add_technicals_indicators(tmp_dataset)
        """ tmp_dataset = self.add_technicals_indicators_sma_rsi_smasignal(tmp_dataset) """
        """ tmp_dataset = self.add_technicals_indicators_sma(tmp_dataset) """
        """ tmp_dataset = self.add_technicals_indicators_rsi(tmp_dataset) """
        """ tmp_dataset = self.add_technicals_indicators_sma_signal(tmp_dataset) """
        # Displaying the entire dataset before price transformation :
        display_results = DisplayResultsService()
        display_results.display_all_dataset(tmp_dataset)
        # Adding lags if specified :
        if lags is not None:
            tmp_dataset = self.add_lag_features(tmp_dataset, lags)
        # Adding historical volatility if specified :
        if add_volatility:
            tmp_dataset = self.calculate_historical_volatility(tmp_dataset)
        # Subsampling :
        tmp_dataset = self.subsample_old_data(tmp_dataset, cutoff_date, fraction=0.1)
        # Normalization :
        tmp_dataset_copy = tmp_dataset.copy()
        columns_to_normalize = ['Dernier']
        scaler = self.get_fitted_scaler(tmp_dataset_copy[columns_to_normalize])
        model_dataset = tmp_dataset
        normalized_datas = self.normalize_datas(tmp_dataset_copy[columns_to_normalize], scaler)
        model_dataset[columns_to_normalize] = normalized_datas
        dates = model_dataset['Date']
        # Removal of the date column :
        del model_dataset['Date']
        # Saving dataset :
        self.save_tmp_dataset(model_dataset)
        # Creation of training and test datasets :
        train_data, test_data = self.create_train_and_test_dataset(model_dataset)
        time_step = 15
        x_train, y_train = self.create_dataset(train_data, time_step)
        x_test, y_test = self.create_dataset(test_data, time_step)
        return x_train, y_train, x_test, y_test, test_data, dates, scaler



    def prepare_one_dimension_dataset(self, dataset, cutoff_date = '2020-01-01'):
        """ Preparation of the one-dimensional dataset before training """
        # Preparation of the dataset :
        tmp_dataset = self.format_dataset(dataset)
        tmp_dataset = self.delete_columns(tmp_dataset)
        # Displaying the entire dataset before price transformation :
        display_results = DisplayResultsService()
        display_results.display_all_dataset(tmp_dataset)
        # Subsampling :
        tmp_dataset = self.subsample_old_data(tmp_dataset, cutoff_date, fraction=0.1)
        # Normalization :
        tmp_dataset_copy = tmp_dataset.copy()
        columns_to_normalize = ['Dernier']
        scaler = self.get_fitted_scaler(tmp_dataset_copy[columns_to_normalize])
        model_dataset = tmp_dataset
        normalized_datas = self.normalize_datas(tmp_dataset_copy[columns_to_normalize], scaler)
        model_dataset[columns_to_normalize] = normalized_datas
        dates = model_dataset['Date']
        # Removal of the date column :
        del model_dataset['Date']
        # Saving the dataset :
        self.save_tmp_dataset(model_dataset)
        # Creation of training and test datasets :
        train_data, test_data = self.create_train_and_test_dataset(model_dataset)
        time_step = 15
        x_train, y_train = self.create_dataset(train_data, time_step)
        x_test, y_test = self.create_dataset(test_data, time_step)
        return x_train, y_train, x_test, y_test, test_data, dates, scaler










    """ ************** Others versions of add_technicals_indicators() method ************** """



    def add_technicals_indicators_sma(self, tmp_dataset):
        """ Adding technical indicators to the dataset (SMA, SMA crossings, and RSI) """
        # Adding indicators in the columns :
        tmp_dataset['MA_150'] = TechnicalIndicatorsService.ma(tmp_dataset, 150)
        tmp_dataset['MA_100'] = TechnicalIndicatorsService.ma(tmp_dataset, 100)
        tmp_dataset['MA_50'] = TechnicalIndicatorsService.ma(tmp_dataset, 50)
        # Remove rows where MA_150 is NaN :
        tmp_dataset = tmp_dataset.dropna(subset=['MA_150'])
        return tmp_dataset



    def add_technicals_indicators_sma_rsi_smasignal(self, tmp_dataset):
        """ Adding technical indicators to the dataset (removal of SMA crossings and RSI but keeping SMA) """
        # Adding indicators in the columns :
        tmp_dataset['MA_150'] = TechnicalIndicatorsService.ma(tmp_dataset, 150)
        tmp_dataset['MA_100'] = TechnicalIndicatorsService.ma(tmp_dataset, 100)
        tmp_dataset['MA_50'] = TechnicalIndicatorsService.ma(tmp_dataset, 50)
        tmp_dataset['RSI'] = TechnicalIndicatorsService.rsi(tmp_dataset, 14)
        # Adding signals generated by the indicators :
        TechnicalIndicatorsService.calculate_signal(tmp_dataset, 50, 150)
        TechnicalIndicatorsService.calculate_signal(tmp_dataset, 100, 150)
        TechnicalIndicatorsService.calculate_signal(tmp_dataset, 50, 100)
        # Remove rows where MA_150 is NaN :
        tmp_dataset = tmp_dataset.dropna(subset=['MA_150'])
        return tmp_dataset



    def add_technicals_indicators_rsi_smasignal(self, tmp_dataset):
        """ Adding technical indicators to the dataset (removal of SMA but keeping SMA crossings and RSI) """
        # Adding indicators in the columns :
        tmp_dataset['RSI'] = TechnicalIndicatorsService.rsi(tmp_dataset, 14)
        # Adding signals generated by the indicators :
        TechnicalIndicatorsService.calculate_signal(tmp_dataset, 50, 150)
        TechnicalIndicatorsService.calculate_signal(tmp_dataset, 100, 150)
        TechnicalIndicatorsService.calculate_signal(tmp_dataset, 50, 100)
        return tmp_dataset



    def add_technicals_indicators_rsi(self, tmp_dataset):
        """ Adding technical indicators to the dataset (removal of SMA, SMA crossings but keeping RSI) """
        tmp_dataset['RSI'] = TechnicalIndicatorsService.rsi(tmp_dataset, 14)
        return tmp_dataset



    def add_technicals_indicators_sma_signal(self, tmp_dataset):
        """ Adding technical indicators to the dataset (removal of SMA, RSI but keeping SMA crossings) """
        # Adding signals generated by the indicators :
        TechnicalIndicatorsService.calculate_signal(tmp_dataset, 50, 150)
        TechnicalIndicatorsService.calculate_signal(tmp_dataset, 100, 150)
        TechnicalIndicatorsService.calculate_signal(tmp_dataset, 50, 100)
        return tmp_dataset


































""" ******************* Traduction des commentaires ******************* """




""" Performs processing on the dataset before training the model """

""" Data preparation """

""" Removal of columns from the original dataset """

""" Adding technical indicators to the dataset (without adding SMA crossings) """
# Adding indicators in the columns:
# Remove rows where MA_150 is NaN:

""" Adjusts the scaler """

""" Normalizes the dataset """

""" Creation of training and test datasets """

""" Generates multi-dimensional training and test datasets """

""" Generates training and test datasets """

""" Creation of data matrices """
        # Creating datasets using the create_dataset function:
        # Displaying the dimensions of the datasets:

""" Generates training and test datasets for a matrix """

""" Subsampling of old data """

""" Adding lags """

""" Calculation of historical volatility """
# Calculation of logarithmic returns
# Calculation of historical volatility
# Removal of the temporary 'Returns' column

""" Saving the dataset to a .csv file """

""" Preparation of the multi-dimensional dataset before training """
# Preparation of the dataset:
# Saving the formatted dataset:
# Adding technical indicators to the dataset:
# Displaying the entire dataset before price transformation:
# Adding lags if specified:
# Adding historical volatility if specified:
# Subsampling:
# Normalization:
# Removal of the date column:
# Saving the dataset:
# Creation of training and test datasets:

""" Preparation of the one-dimensional dataset before training """
# Preparation of the dataset:
# Displaying the entire dataset before price transformation:
# Subsampling:
# Normalization:
# Removal of the date column:
# Saving the dataset:
# Creation of training and test datasets:

""" Adding technical indicators to the dataset (SMA, SMA crossings, and RSI) """
# Adding indicators in the columns:
# Remove rows where MA_150 is NaN

""" Adding technical indicators to the dataset (removal of SMA crossings and RSI but keeping SMA) """
# Adding indicators in the columns:
# Adding signals generated by the indicators:
# Remove rows where MA_150 is NaN

""" Adding technical indicators to the dataset (removal of SMA but keeping SMA crossings and RSI) """
# Adding indicators in the columns:
# Adding signals generated by the indicators:

""" Adding technical indicators to the dataset (removal of SMA, SMA crossings but keeping RSI) """

""" Adding technical indicators to the dataset (removal of SMA, RSI but keeping SMA crossings) """
# Adding signals generated by the indicators:
