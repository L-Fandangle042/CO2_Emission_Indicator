import os
from os.path import join
import numpy as np
import pandas as pd
import seaborn as sns
import pmdarima as pm
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.base.prediction import PredictionResults

from api.Country import Country


class ArimaTrainer:

    raw_data_path = join('..','data','carbon_dioxide','CO2_YEARLY_DATA_1970-2021.xlsx')
    preprocessed_data_path = join('..','data','carbon_dioxide','CO2_YEARLY_preprocessed.csv')

    def preprocess(self) -> DataFrame:
        data = pd.read_excel(self.raw_data_path, sheet_name="TOTALS BY COUNTRY", skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                             header=1)

        data = data.drop(['IPCC_annex', 'IPCC_annex', 'Country_code_A3', 'C_group_IM24_sh', 'Substance'], axis=1)
        data.columns = [c.removeprefix("Y_") for c in data.columns]

        countries_of_interest = data['Name'].tolist()
        data = data[data['Name'].isin(countries_of_interest)].head(len(countries_of_interest))
        data = data.reset_index(drop=True)

        df = pd.DataFrame(np.array(data.columns[2:], dtype='int16'), columns=['year'])

        for i in range(len(countries_of_interest)):
            name: str = str(data.at[i, 'Name'])
            df[name] = data.iloc[i, 2:].values

        df['year'] = pd.to_datetime(df['year'], format='%Y')
        df.set_index('year', inplace=True)

        df.to_csv(self.preprocessed_data_path, index=True)

        return df

    @staticmethod
    def plot_prediction(values: ndarray):
        df_prediction = pd.DataFrame(np.array(values, dtype='float32'), columns=['CO2_volume'])
        df_prediction['year'] = pd.date_range(start='1970-01-01', periods=len(df_prediction), freq='Y')

        sns.lineplot(data=df_prediction, x='year', y='CO2_volume')
        plt.show()

    def execute(self, country: Country, horizon: int, environment: str = 'cloud') -> bool:

        if os.path.exists(self.preprocessed_data_path):
            df = pd.read_csv(self.preprocessed_data_path)
        else:
            df = self.preprocess()

        horizon = horizon - int(df['year'].max()[:4])

        trend = None

        auto_defined_parameters = pm.auto_arima(df[country],
                                                max_order=10,
                                                trend=trend,
                                                seasonal=False)

        order = auto_defined_parameters.get_params().get("order")

        arima = ARIMA(df[country], order=order, trend=trend).fit()

        forecast_results: PredictionResults = arima.get_forecast(horizon)

        historic_plus_predicted_values: ndarray = np.append(df[country].values, forecast_results.predicted_mean.values)

        if environment == 'local':
            self.plot_prediction(historic_plus_predicted_values)

        target_reduction = 0.45
        target = df[country][39] * (1 - target_reduction)

        return bool(historic_plus_predicted_values[-1] <= target)
