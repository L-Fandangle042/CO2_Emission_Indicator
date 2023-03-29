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
# async def predict(horizon: int = 9,
#                   learning_rate: float = 0.001,
#                   lstm_units: int = 50,
#                   first_dense_layer: int = 32,
#                   second_dense_layer: int = 16,
#                   loss: str = 'mse',
#                   patience: int = 5,
#                   validation_split: float = 0.25,
#                   batch_size: int = 32,
#                   epochs: int = 100,
#                   dropout_rate: float = 0.2,
#                   add_second_layer: bool = False,
#                   env: str = 'cloud') -> bool:
#     return RnnTrainer().execute(horizon=horizon,
#                                 learning_rate=learning_rate,
#                                 lstm_units=lstm_units,
#                                 first_dense_layer=first_dense_layer,
#                                 second_dense_layer=second_dense_layer,
#                                 loss=loss,
#                                 patience=patience,
#                                 validation_split=validation_split,
#                                 batch_size=batch_size,
#                                 epochs=epochs,
#                                 dropout_rate=dropout_rate,
#                                 add_second_layer=add_second_layer,
#                                 env=env)


@api.get("/predict/rnn")
async def predict(country_code: str = 'FRA', horizon: int = 9) -> bool:
    return RnnPredictor().predict(country_code, horizon, 'cloud')
