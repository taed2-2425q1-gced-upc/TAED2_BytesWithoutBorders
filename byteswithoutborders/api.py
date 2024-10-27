"""Main script: it includes our API initialization and endpoints."""

import logging
import pickle
import torch
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Dict, Union
from pydantic import BaseModel

import tensorflow as tf
import tensorflow_hub as hub
from codecarbon import track_emissions
from fastapi import FastAPI, HTTPException, UploadFile, File


class ModelSingleInput(BaseModel):
    image_data: list


class ModelDoubleInput(BaseModel):
    image_data_1: list
    image_data_2: list


global model

app = FastAPI(
    title="Classifying fashion",
    description="This API lets you make predictions about clothing items",
    version="0.1",
)


@app.on_event("startup")
async def startup_event():
    #Load the TensorFlow model when the application starts.
    global model
    model = tf.keras.models.load_model("../models/fashion-model.h5", compile=False)


@app.on_event("shutdown")
async def shutdown_event():
    #Clear the model from memory when the application stops.
    global model
    model = None


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Fashion Classifier!"}

# Decode the image to a tensor, convert to grayscale (1 channel)
# Resize the image to 28x28
# Reshape to add batch dimension and a single color channel
async def process_image(file: UploadFile) -> tf.Tensor:
    image_bytes = await file.read()
    image = tf.io.decode_image(image_bytes, channels=1)
    image = tf.image.resize(image, [28, 28])
    image = tf.reshape(image, [1, 28, 28, 1])
    return image


@track_emissions(
    project_name="clothing-item-prediction",
    measure_power_secs=1,
    save_to_file=True,
    output_dir="metrics"
)
@app.post("/predict/single-image/uploadFile", tags=["Prediction"])
async def _predict_single_image(file: UploadFile = File(...)):
    try:
        image = await process_image(file)

        prediction = model.predict(image)
        predicted_class = tf.argmax(prediction, axis=1).numpy()[0]

        return {"predicted_class": int(predicted_class)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@track_emissions(
    project_name="clothing-item-prediction",
    measure_power_secs=1,
    save_to_file=True,
    output_dir="metrics"
)
@app.post("/predict/double-image/uploadFile", tags=["Prediction"])
async def _predict_double_image(file_1: UploadFile = File(...), file_2: UploadFile = File(...)):
    try:

        image_1 = await process_image(file_1)
        image_2 = await process_image(file_2)

        prediction_1 = model.predict(image_1)
        predicted_class_1 = tf.argmax(prediction_1, axis=1).numpy()[0]

        prediction_2 = model.predict(image_2)
        predicted_class_2 = tf.argmax(prediction_2, axis=1).numpy()[0]

        return {
            "predicted_class_image_1": int(predicted_class_1),
            "predicted_class_image_2": int(predicted_class_2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@track_emissions(
    project_name="clothing-item-prediction",
    measure_power_secs=1,
    save_to_file=True,
    output_dir="metrics"
)
@app.post("/predict/single-image/raw", tags=["Prediction"])
def _predict_single_image(input: ModelSingleInput):
    try:
        # Convert input list to Tensor and reshape for model
        data = tf.convert_to_tensor([input.image_data])
        # reshape to 28x28 grayscale image
        data = tf.reshape(data, [1, 28, 28, 1])

        # Make predictions
        prediction = model.predict(data)
        predicted_class = tf.argmax(prediction, axis=1).numpy()[0]
        return {"predicted_class": int(predicted_class)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@track_emissions(
    project_name="clothing-item-prediction",
    measure_power_secs=1,
    save_to_file=True,
    output_dir="metrics"
)
@app.post("/predict/double-image/raw", tags=["Prediction"])
def _predict_double_image(input: ModelDoubleInput):
    try:
        # Process the first image
        data_1 = tf.convert_to_tensor([input.image_data_1])
        data_1 = tf.reshape(data_1, [1, 28, 28, 1])  # reshape to 28x28 grayscale image

        # Process the second image
        data_2 = tf.convert_to_tensor([input.image_data_2])
        data_2 = tf.reshape(data_2, [1, 28, 28, 1])  # reshape to 28x28 grayscale image

        # Make predictions for both images
        prediction_1 = model.predict(data_1)
        predicted_class_1 = tf.argmax(prediction_1, axis=1).numpy()[0]

        prediction_2 = model.predict(data_2)
        predicted_class_2 = tf.argmax(prediction_2, axis=1).numpy()[0]

        # Return predictions for both images
        return {
            "predicted_class_image_1": int(predicted_class_1),
            "predicted_class_image_2": int(predicted_class_2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
