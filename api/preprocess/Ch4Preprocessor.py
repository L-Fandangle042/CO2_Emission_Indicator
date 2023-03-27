from api.preprocess.Preprocessor import Preprocessor

from pandas import DataFrame


class Ch4Preprocessor(Preprocessor):

    raw_data_path = 'data/methane/CH4_YEARLY_DATA_1970-2021.xlsx'
    preprocessed_data_path = 'data/methane/CH4_YEARLY_preprocessed.csv'

    def preprocess(self) -> DataFrame:
        return self.common()
