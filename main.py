import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from schemas import Customer
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://127.0.0.1:5500/"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("model.pkl", 'rb') as file:
    model = pickle.load(file)

@app.get("/")
def read_root():
    return FileResponse("index.html")
# app.mount("/static", StaticFiles(directory="static"), name="static")

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