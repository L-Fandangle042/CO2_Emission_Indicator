from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from pandas import DataFrame


class Preprocessor(ABC):

    raw_data_path = ''
    preprocessed_data_path = ''

    def get_raw_data_path(self):
        return self.raw_data_path

    def get_preprocessed_data_path(self):
        return self.preprocessed_data_path

    @abstractmethod
    def preprocess(self):
        pass

    def common(self) -> DataFrame:
        data = pd.read_excel(self.get_raw_data_path(),
                             sheet_name="TOTALS BY COUNTRY",
                             skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                             header=1)

        data = data.drop(
            ['IPCC_annex', 'IPCC_annex', 'Country_code_A3', 'C_group_IM24_sh',
             'Substance'], axis=1)
        data.columns = [c.removeprefix("Y_") for c in data.columns]

        countries_of_interest = data['Name'].tolist()
        data = data[data['Name'].isin(countries_of_interest)].head(
            len(countries_of_interest))
        data = data.reset_index(drop=True)

        df = pd.DataFrame(np.array(data.columns[2:], dtype='int16'),
                          columns=['year'])

        for i in range(len(countries_of_interest)):
            name: str = str(data.at[i, 'Name'])
            df[name] = data.iloc[i, 2:].values

        df['year'] = pd.to_datetime(df['year'], format='%Y')
        df.set_index('year', inplace=True)

        df.to_csv(self.get_preprocessed_data_path(), index=True)

        return df
