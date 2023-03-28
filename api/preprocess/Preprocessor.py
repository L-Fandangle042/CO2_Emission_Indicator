import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from pandas import DataFrame


class Preprocessor(ABC):

    prefix = ''
    raw_data_path = ''
    preprocessed_data_path = ''

    def get_prefix(self):
        return self.prefix

    def get_raw_data_path(self):
        return self.raw_data_path

    def get_preprocessed_data_path(self):
        return self.preprocessed_data_path

    @abstractmethod
    def preprocess(self):
        pass

    @classmethod
    def to_datetime_year(cls, df, col_name):
        df[col_name] = pd.to_datetime(df[col_name], format='%Y')
        df.set_index(col_name, inplace=True)
        return df

    def preprocess_gases(self) -> DataFrame:

        if os.path.exists(self.get_preprocessed_data_path()):
            return pd.read_csv(self.get_preprocessed_data_path())
        else:
            data = pd.read_excel(self.get_raw_data_path(),
                                 sheet_name="TOTALS BY COUNTRY",
                                 skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                                 header=1)

            data = data.drop(
                ['IPCC_annex', 'IPCC_annex', 'Name', 'C_group_IM24_sh',
                 'Substance'], axis=1)
            data.columns = [c.removeprefix("Y_") for c in data.columns]

            countries_of_interest = data['Country_code_A3'].tolist()
            data = data[data['Country_code_A3'].isin(countries_of_interest)]\
                .head(len(countries_of_interest))
            data = data.reset_index(drop=True)

            df = pd.DataFrame(np.array(data.columns[2:],
                                       dtype='int16'), columns=['year'])

            for i in range(len(countries_of_interest)):
                name: str = f"{self.get_prefix()}_{str(data.at[i, 'Country_code_A3'])}"
                df[name] = data.iloc[i, 2:].values

            df = Preprocessor.to_datetime_year(df, 'year')

            df.to_csv(self.get_preprocessed_data_path(), index=True)

            return df
