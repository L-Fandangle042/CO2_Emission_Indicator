  # Instead of creating an image from scratch, we use this image which has python installed.
FROM python:3.8.6-buster


# COPY allows you to select the folders and files to include in your docker image
# Here, we will include our api_folder and the requiremenets.txt file
COPY api /api
COPY requirements.txt /requirements.txt
COPY data /data

# RUN allows you to run terminal commands when your image gets created
# Here, we upgrade pip and install the libraries in our requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt



# CMD controls the functionality of the container
# Here, we use uvicorn to control the web server ports

# local
#CMD uvicorn api.api:api --host 0.0.0.0

# deploy to gcp
CMD uvicorn api.api:api --host 0.0.0.0 --port $PORT
