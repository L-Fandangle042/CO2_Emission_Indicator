from fastapi import FastAPI

from api.ArimaTrainer import ArimaTrainer
from api.Country import Country
from api.RnnPredictor import RnnPredictor
from api.RnnTrainer import RnnTrainer


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


# closed for online training
# @api.get("/train/rnn")
# async def predict(horizon: int = 9) -> bool:
#     return RnnTrainer().execute(horizon)


@api.get("/predict/rnn")
async def predict(country_code: str = 'FRA', horizon: int = 9) -> bool:
    return RnnPredictor().predict(country_code, horizon, 'cloud')
