from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd

app = FastAPI()

# Load the trained model
model = joblib.load(r"C:\Users\fady\Downloads\Machine Project\naive_bayes_model.joblib")

# Instantiate a LabelEncoder
label_encoder = LabelEncoder()

# CORS (Cross-Origin Resource Sharing) middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for security in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a Pydantic model for the input data
class InputData(BaseModel):
    date: str
    location: str
    minimum_temp: float
    maximum_temp: float
    rain_fall: float
    Wind_gustspeed: int
    Winddir_9am: str
    Winddir_3pm: str
    Wind_speed9am: int
    Wind_speed3pm: int
    Humidity_9am: int
    Humidity_3pm: int
    Pressure_9am: float
    Pressure_3pm: float
    Cloud_9am: int
    cloud3pm: int
    Temp_9am: float
    Temp_3pm: float
    Rain_today: int
    risk_mm: int

# API endpoint for making predictions
@app.post('/predict')
async def predict(input_data: InputData):
    try:
        # Convert Pydantic model to a DataFrame
        df = pd.DataFrame([input_data.dict()])

        # Label encode categorical columns
        categorical_columns = ["location", "Winddir_9am", "Winddir_3pm", "date"]
        for col in categorical_columns:
            df[col] = label_encoder.fit_transform(df[col])

        # Make predictions using the loaded model
        predictions = model.predict(df)

        # Return the predictions as JSON
        return {'predictions': predictions.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)