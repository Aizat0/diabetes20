from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn


diabetes_app = FastAPI()

scaler = joblib.load('scaler (4).pkl')
model = joblib.load('model (3).pkl')


class DiabetesSchema(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


@diabetes_app.post('/')
async def diabetes_predicted(diabetes: DiabetesSchema):
    diabetes_dict = diabetes.dict()


    features = list(diabetes_dict.values())

    scaled_data = scaler.transform([features])
    pred = model.predict_proba(scaled_data)[0]
    probability = float(pred[1])
    if probability > 0.5:
        diabetes_final = "Yes"
    else:
        diabetes_final = "No"

    return {
        "diabetes": diabetes_final,
        "probability": round(probability)
    }



if __name__ == '__main__':
    uvicorn.run(diabetes_app, host='127.0.0.1', port=8001)