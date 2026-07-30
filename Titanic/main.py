from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


app = FastAPI(
    title="Titanic Survival Prediction API"
)


model = joblib.load("titanic_model.pkl")
scaler = joblib.load("titanic_scaler.pkl")



class Passenger(BaseModel):

    pclass: int
    sex: int
    age: int
    sibsp: int
    parch: int
    fare: int
    embarked: int
    alone: int



@app.get("/")
def home():

    return {
        "message": "Titanic API Running"
    }



@app.post("/predict")
def predict(data: Passenger):

    input_data = np.array(
        [[
            data.pclass,
            data.sex,
            data.age,
            data.sibsp,
            data.parch,
            data.fare,
            data.embarked,
            data.alone
        ]]
    )


    input_scaled = scaler.transform(input_data)


    prediction = model.predict(input_scaled)[0]


    result = "Survived" if prediction == 1 else "Not Survived"


    return {
    "input": {
        "pclass": data.pclass,
        "sex": data.sex,
        "age": data.age,
        "sibsp": data.sibsp,
        "parch": data.parch,
        "fare": data.fare,
        "embarked": data.embarked,
        "alone": data.alone
    },
    "prediction": int(prediction),
    "result": result
}