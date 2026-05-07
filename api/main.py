import os
import pickle
import pandas as pd
from schemas import Customer
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*",  ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.post("/predict")
def predict(customer : Customer ):
    try:
        data = pd.DataFrame([[
        customer.age, 
        customer.sex, 
        customer.bmi,  # This is computed from height/weight
        customer.children, 
        customer.smoker, 
        customer.region
        ]], columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
    
        prediction = model.predict(data)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Mount static files at the end so they don't interfere with API routes
public_path = os.path.join(os.path.dirname(__file__), "../public")
if os.path.exists(public_path):
    app.mount("/", StaticFiles(directory=public_path, html=True), name="static")