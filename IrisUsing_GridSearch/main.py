from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="Iris Flower Classification API")


model = pickle.load(open("iris_svm_model.pkl", "rb"))


class IrisInput(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {"message": "Iris Flower Classification API is running"}


@app.post("/predict")
def predict(data: IrisInput):

    flower_data = np.array(data.features).reshape(1, -1)

    prediction = model.predict(flower_data)[0]

    return {
        "input": data.features,
        "prediction": prediction
    }