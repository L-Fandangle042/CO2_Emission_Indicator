from api.preprocess.Preprocessor import Preprocessor

from pandas import DataFrame


class Ch4Preprocessor(Preprocessor):

    prefix = 'CH4'
    raw_data_path = 'data/methane/CH4_YEARLY_DATA_1970-2021.xlsx'
    preprocessed_data_path = 'data/methane/CH4_YEARLY_preprocessed_cc.csv'

    def preprocess(self) -> DataFrame:
        return self.preprocess_gases()
