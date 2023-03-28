from api.preprocess.Preprocessor import Preprocessor

from pandas import DataFrame


class N2oPreprocessor(Preprocessor):

    prefix = 'N2O'
    raw_data_path = 'data/nitrogen_dioxide/N2O_YEARLY_DATA_1970-2021.xlsx'
    preprocessed_data_path = 'data/nitrogen_dioxide/N2O_YEARLY_preprocessed_cc.csv'

    def preprocess(self) -> DataFrame:
        return self.preprocess_gases()
