import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import load_model
from numpy import ndarray
from pandas import DataFrame


class RnnPredictor:

    get_preprocessed_data_path_no_gdp = 'data/preprocessed_no_gdp_cc.csv'

    def predict(self, country_code: str, horizon: int = 9, env: str = 'cloud') -> bool:

        match horizon:
            case 9:
                model = load_model('models/lstm_gases_2030.h5')
            case 29:
                model = load_model('models/lstm_gases_2050.h5')
            case _:
                raise ValueError('Horizon not supported')

        initial_df: DataFrame = pd.read_csv(self.get_preprocessed_data_path_no_gdp)

        target = 'CO2'
        test_country = country_code

        columns = [c for c in initial_df.columns if c.endswith(test_country)]

        df = initial_df.loc[horizon:, columns]

        for c in columns:
            if c == f'{target}_{test_country}':
                mean = df[c].mean()
                std = df[c].std()
            df[c] = (df[c] - df[c].mean()) / df[c].std()

        X = df.values.reshape(1, df.shape[0], df.shape[1])

        predictions: ndarray = model.predict(X).ravel()

        predictions = predictions * std + mean
        actual = initial_df[f'{target}_{test_country}'].values

        df_prediction = pd.DataFrame(np.concatenate((actual, predictions), axis=0),
                                     columns=['CO2_predicted'])
        df_prediction['CO2_actual'] = np.concatenate((actual,
                                                      np.zeros(len(predictions))),
                                                     axis=0)
        df_prediction['year'] = pd.date_range(start='1970-01-01',
                                              periods=len(df_prediction),
                                              freq='Y')

        target_reduction = 0.45
        target = actual[39] * (1 - target_reduction)

        if env == 'local':
            sns.lineplot(data=df_prediction)
            plt.show()
            print(int(target), int(predictions[-1]))

        return bool(predictions[-1] <= target)
