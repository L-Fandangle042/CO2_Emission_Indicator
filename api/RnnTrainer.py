import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers
from numpy import ndarray
from pandas import DataFrame

from api.preprocess.Ch4Preprocessor import Ch4Preprocessor
from api.preprocess.Co2Preprocessor import Co2Preprocessor
from api.preprocess.GdpPreprocessor import GdpPreprocessor
from api.preprocess.N2oPreprocessor import N2oPreprocessor

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 800)


class RnnTrainer:

    get_preprocessed_data_path_no_gdp = 'data/preprocessed_no_gdp_cc.csv'
    get_preprocessed_data_path = 'data/preprocessed_cc.csv'

    @classmethod
    def get_x_and_y(cls, df: DataFrame, countries: [str], n_SEQUENCES: int,
                    n_OBSERVATIONS: int, target: str, horizon: int) -> tuple:

        columns: [str] = df.columns

        sequence = []
        targets = []

        for country_code in countries:
            country_columns: [str] = [column for column in columns if column.split('_')[-1] == country_code]
            if len(country_columns) == n_SEQUENCES:
                country_df: DataFrame = df[country_columns]
                train_sequence = country_df[:n_OBSERVATIONS].values
                # print(train_sequence)
                sequence.append(train_sequence)

                target_column = [c for c in country_df.columns if c.startswith(target)][0]
                target_sequence = country_df[target_column].values[-horizon:]
                # print(country_df[target_column].values)
                # print(country_df[target_column].values[-horizon:])
                targets.append(target_sequence)

        X = np.array(sequence)
        y = np.expand_dims(np.array(targets), axis=-1)

        return X, y

    def execute(self, with_gdp: str) -> bool:

        match with_gdp:
            case 'with_gdp':
                n_SEQUENCES = 5
                if os.path.exists(self.get_preprocessed_data_path):
                    df = pd.read_csv(self.get_preprocessed_data_path)
                else:
                    co2: DataFrame = Co2Preprocessor().preprocess()
                    ch4: DataFrame = Ch4Preprocessor().preprocess()
                    n2o: DataFrame = N2oPreprocessor().preprocess()
                    gdp: DataFrame = GdpPreprocessor().preprocess()
                    gdp.fillna(0, inplace=True)

                    df: DataFrame = pd.merge(co2, ch4, on='year')
                    df: DataFrame = pd.merge(df, n2o, on='year')
                    df: DataFrame = pd.merge(df, gdp, on='year')

                    df.to_csv(self.get_preprocessed_data_path, index=False)

            case 'gases':
                n_SEQUENCES = 3
                if os.path.exists(self.get_preprocessed_data_path_no_gdp):
                    df = pd.read_csv(self.get_preprocessed_data_path_no_gdp)
                else:
                    co2: DataFrame = Co2Preprocessor().preprocess()
                    ch4: DataFrame = Ch4Preprocessor().preprocess()
                    n2o: DataFrame = N2oPreprocessor().preprocess()

                    df: DataFrame = pd.merge(co2, ch4, on='year')
                    df: DataFrame = pd.merge(df, n2o, on='year')

                    df.to_csv(self.get_preprocessed_data_path_no_gdp, index=False)

            case 'co2':
                n_SEQUENCES = 1
                df: DataFrame = Co2Preprocessor().preprocess()

        year = 'year'
        target = 'CO2'
        test_country = 'JPN'

        HORIZON = 9
        n_OBSERVATIONS = df.shape[0] - HORIZON

        for c in df.columns[1:]:
            if c == f'CO2_{test_country}':
                mean = df[c].mean()
                std = df[c].std()
            df[c] = (df[c] - df[c].mean()) / df[c].std()

        df = df.drop([year], axis=1)
        columns: [str] = df.columns[1:]
        countries_codes: [str] = sorted(set([column.split('_')[-1] for column in columns]))

        train_test_ratio: float = 0.8
        split_point: int = int(len(countries_codes) * train_test_ratio)
        # train_countries: [str] = countries_codes[:split_point]
        train_countries: [str] = [c for c in countries_codes if c != test_country]
        # test_countries: [str] = countries_codes[split_point:]
        test_countries: [str] = [test_country]

        X_train, y_train = self.get_x_and_y(df, train_countries, n_SEQUENCES, n_OBSERVATIONS, target, HORIZON)

        model = Sequential()
        model.add(layers.LSTM(units=50, activation='tanh', input_shape=(n_OBSERVATIONS, n_SEQUENCES)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(layers.Dense(HORIZON, activation="linear"))

        model.compile(loss='mse', optimizer=Adam(learning_rate=0.004), metrics=['mae'])

        model.fit(X_train, y_train, validation_split=0.2, batch_size=32, epochs=100, verbose=1)

        X_test, y_test = self.get_x_and_y(df, test_countries, n_SEQUENCES, n_OBSERVATIONS, target, HORIZON)

        print(X_train.shape, X_test.shape)

        predictions: ndarray = model.predict(X_test).ravel()

        predictions = predictions * std + mean
        actual = df[f'CO2_{test_country}'].values * std + mean

        df_prediction = pd.DataFrame(np.array(actual, dtype='float32'),
                                     columns=['CO2_actual'])
        print(predictions, actual[n_OBSERVATIONS:])
        df_prediction['CO2_predicted'] = np.concatenate((actual[:n_OBSERVATIONS], predictions), axis=0)
        df_prediction['year'] = pd.date_range(start='1970-01-01',
                                              periods=len(df_prediction),
                                              freq='Y')

        sns.lineplot(data=df_prediction)
        plt.show()

        print(abs(actual[-1] - predictions[-1]) / actual[-1])

        target_reduction = 0.45
        target = actual[39] * (1 - target_reduction)

        print(actual)
        print(target, predictions[-1])

        return bool(predictions[-1] <= target)
