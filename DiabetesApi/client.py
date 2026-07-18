import requests

url = "http://127.0.0.1:8000/diabetes_pred"

input_data_for_model = {
    "pregnancies": 5,
    "Glucose": 166,
    "BloodPressure": 72,
    "SkinThickness": 19,
    "Insulin": 175,
    "BMI": 25.8,
    "DiabetesPedigreeFunction": 0.587,
    "Age": 51
}

response = requests.post(url, json=input_data_for_model)

print(response.text)