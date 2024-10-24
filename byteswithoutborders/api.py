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
from fastapi import FastAPI, HTTPException, UploadFile



class ModelInput(BaseModel):
    image_data: list


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

@track_emissions(
    project_name="clothing-item-prediction",
    measure_power_secs=1,
    save_to_file=True,
    output_dir="metrics"
)
@app.post("/predict/single-image", tags=["Prediction"])
def _predict_single_image(input: ModelInput):
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



