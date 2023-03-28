import os

import numpy as np
import pandas as pd

from api.preprocess.Preprocessor import Preprocessor

from pandas import DataFrame


class GdpPreprocessor(Preprocessor):

    prefix = 'GDP'
    raw_data_path = 'data/world_country_gdp_usd.csv'
    preprocessed_data_path = 'data/world_country_gdp_preprocessed_cc.csv'

    def preprocess(self) -> DataFrame:

        if os.path.exists(self.get_preprocessed_data_path()):
            return pd.read_csv(self.get_preprocessed_data_path())
        else:
            df = pd.read_csv(self.get_raw_data_path())

            df = df.drop(['Country Name'], axis=1)

            countries = df['Country Code'].unique()
            years = sorted(df['year'].unique())

            new_df = pd.DataFrame(np.array(years, dtype='int16'), columns=['year'])

            for country in countries:
                gdp = df[df['Country Code'] == country]['GDP_USD'].values
                gdp_per_capita = df[df['Country Code'] == country]['GDP_per_capita_USD'].values
                new_df[f'{self.prefix}_{country}'] = gdp
                new_df[f'{self.prefix}_PC_{country}'] = gdp_per_capita

            new_df = new_df[new_df['year'] >= 1971]

            new_df = Preprocessor.to_datetime_year(new_df, 'year')

            new_df.to_csv(self.get_preprocessed_data_path(), index=True)

            return new_df


