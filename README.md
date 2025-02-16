# CO₂ Emissions Forecasting API 🚀

This project provides a **CO₂ emissions forecasting system** using **ARIMA time series modeling**.  
It includes an **API built with FastAPI** that allows users to request emissions predictions for different countries.

## 📌 Features
- **FastAPI-based API** to interact with emissions data and predictions.
- **Data preprocessing** from raw Excel files containing CO₂ emissions data.
- **Time series forecasting** using **ARIMA models** (Auto-ARIMA for parameter selection).
- **Visualization**.

## 🔧 Installation
### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/co2-forecasting-api.git
```
### 2️⃣ Install Dependencies
> pip install -r requirements.txt

### 3️⃣ Run the API Server
> uvicorn main:app --reload

API URL: https://co2project-vzzs3rfq7q-ew.a.run.app

# 📊 Forecasting with ARIMA
The ArimaTrainer class handles time series forecasting for CO₂ emissions, predicting emissions for a given country and years

Example Usage:
```
from models.arima_trainer import ArimaTrainer
from api.Country import Country

trainer = ArimaTrainer()
result = trainer.execute(country="USA", horizon=2050, environment="local")

if result:
    print("Emissions target met!")
else:
    print("Target not met, further reductions needed.")
```

---
### GCP
```
export GCP_PROJECT_ID="co2indicator"
export DOCKER_IMAGE_NAME="co2project"
export GCR_MULTI_REGION="eu.gcr.io"
export GCR_REGION="europe-west1"

docker build -t $GCR_MULTI_REGION/$GCP_PROJECT_ID/$DOCKER_IMAGE_NAME .

docker push $GCR_MULTI_REGION/$GCP_PROJECT_ID/$DOCKER_IMAGE_NAME

gcloud run deploy --image $GCR_MULTI_REGION/$GCP_PROJECT_ID/$DOCKER_IMAGE_NAME --platform managed --region $GCR_REGION
```
