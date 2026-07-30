from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import pickle


app = FastAPI()


# Load model once when server starts
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


with open("kmeans.pkl", "rb") as f:
    model = pickle.load(f)



class CustomerData(BaseModel):
    feature_1: float
    feature_2: float



@app.get("/")
def home():
    return {
        "message": "Clustering API Running"
    }



@app.post("/predict")
def predict(data: CustomerData):

    input_df = pd.DataFrame(
        [[data.feature_1, data.feature_2]],
        columns=[
            "Feature_1",
            "Feature_2"
        ]
    )


    # same scaling
    scaled_data = scaler.transform(input_df)


    # cluster prediction
    cluster = model.predict(scaled_data)


    return {
    "input": {
        "Feature_1": data.feature_1,
        "Feature_2": data.feature_2
    },
    "prediction": {
        "cluster": int(cluster[0])
    }
}