import os

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers
from pandas import DataFrame

from api.preprocess.Ch4Preprocessor import Ch4Preprocessor
from api.preprocess.Co2Preprocessor import Co2Preprocessor
from api.preprocess.N2oPreprocessor import N2oPreprocessor


class RnnTrainer:

    get_preprocessed_data_path_no_gdp = 'data/preprocessed_no_gdp_cc.csv'

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
                sequence.append(train_sequence)

                target_column = [c for c in country_df.columns if c.startswith(target)][0]
                target_sequence = country_df[target_column].values[-horizon:]
                targets.append(target_sequence)

        X = np.array(sequence)
        y = np.expand_dims(np.array(targets), axis=-1)

        return X, y

    def execute(self, horizon: int = 9) -> bool:

        name = 'gases'
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

        assert horizon == 9 or horizon == 29,\
            'Horizon must be 9 or 29 (2030 or 2050)'

        year = 'year'
        target = 'CO2'

        HORIZON = 2030 if horizon == 9 else 2050
        n_OBSERVATIONS = df.shape[0]

        for c in df.columns[1:]:
            df[c] = (df[c] - df[c].mean()) / df[c].std()

        df = df.drop([year], axis=1)
        columns: [str] = df.columns[1:]
        countries_codes: [str] = sorted(set([column.split('_')[-1] for column in columns]))

        train_countries: [str] = [c for c in countries_codes]

        X_train, y_train = self.get_x_and_y(df, train_countries, n_SEQUENCES, n_OBSERVATIONS, target, horizon)

        if horizon == 9:
            model = Sequential()
            model.add(layers.LSTM(units=50, activation='tanh', input_shape=(n_OBSERVATIONS, n_SEQUENCES)))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(16, activation='relu'))
            model.add(layers.Dense(horizon, activation="linear"))
        else:
            model = Sequential()
            model.add(layers.LSTM(units=100, activation='tanh', input_shape=(n_OBSERVATIONS, n_SEQUENCES)))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(layers.Dense(horizon, activation="linear"))

        model.compile(loss='mse', optimizer=Adam(learning_rate=0.004), metrics=['mae'])

        model.fit(X_train, y_train, validation_split=0.2, batch_size=32, epochs=100, verbose=1)

        model.save(f'models/lstm_{name}_{HORIZON}.h5')

        return True
