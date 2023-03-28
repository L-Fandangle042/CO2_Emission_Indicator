import pandas as pd

from api.preprocess.Ch4Preprocessor import Ch4Preprocessor
from api.preprocess.Co2Preprocessor import Co2Preprocessor
from api.preprocess.GdpPreprocessor import GdpPreprocessor
from api.preprocess.N2oPreprocessor import N2oPreprocessor


class RnnTrainer:

    def execute(self) -> bool:

        co2 = Co2Preprocessor().preprocess()
        ch4 = Ch4Preprocessor().preprocess()
        n2o = N2oPreprocessor().preprocess()
        gdp = GdpPreprocessor().preprocess()

        df = pd.concat([co2, ch4, n2o, gdp], axis=1)

        print(df.head(10))

        return True
