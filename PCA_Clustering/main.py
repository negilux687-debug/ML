from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="PCA KMeans Clustering API")


model = pickle.load(open("cluster_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


class ClusterInput(BaseModel):
    features: list


@app.get("/")
def home():
    return {"message": "PCA KMeans Clustering API is running"}


@app.post("/predict")
def predict(data: ClusterInput):

    
    input_data = np.array(data.features).reshape(1, -1)

   
    scaled_data = scaler.transform(input_data)

    cluster = model.predict(scaled_data)[0]

    return {
        "input": data.features,
        "cluster": int(cluster)
    }