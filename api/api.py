from fastapi import FastAPI
from pandas import DataFrame

from api.ArimaTrainer import ArimaTrainer
from api.Country import Country
from api.preprocess.Ch4Preprocessor import Ch4Preprocessor
from api.preprocess.N2oPreprocessor import N2oPreprocessor

api = FastAPI()


# define a root `/` endpoint
@api.get("/")
def index():
    return {"ok": "Hello CO2 team!"}


@api.get("/predict")
async def predict(country: Country = Country.China,
                  max_predicted_year: int = 2030) -> bool:
    assert max_predicted_year > 2022, "max_predicted_year must be greater than 2020"
    assert max_predicted_year < 2050, "max_predicted_year must be less than 2050"

    return ArimaTrainer().execute(country, max_predicted_year, 'cloud')
