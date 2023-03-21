CO2_Emission_Indicator

API URL: https://co2project-vzzs3rfq7q-ew.a.run.app

Stand-up diary: https://docs.google.com/spreadsheets/d/1JJFJJkR3CQ8T_KHsmSxHZcH8993DZDwGjeMQKThae9M/edit#gid=0

# GCP
export GCP_PROJECT_ID="co2indicator"
export DOCKER_IMAGE_NAME="co2project"
export GCR_MULTI_REGION="eu.gcr.io"
export GCR_REGION="europe-west1"

docker build -t $GCR_MULTI_REGION/$GCP_PROJECT_ID/$DOCKER_IMAGE_NAME .

docker push $GCR_MULTI_REGION/$GCP_PROJECT_ID/$DOCKER_IMAGE_NAME

gcloud run deploy --image $GCR_MULTI_REGION/$GCP_PROJECT_ID/$DOCKER_IMAGE_NAME --platform managed --region $GCR_REGION
