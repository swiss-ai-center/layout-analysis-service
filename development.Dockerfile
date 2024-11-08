# Base image
FROM python:3.11

# Install all required packages to run the model
# TODO: 1. Add any additional packages required to run your model

RUN apt update && apt install --yes ffmpeg libsm6 libxext6
