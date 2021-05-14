from fastapi import FastAPI
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from titanic.predict import get_model


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return "Root - Titanic survivor prediction"

@app.get("/survivor_predict")
def survivor_predict(Sex, Age, Fare, Pclass):
    model = get_model("model")
    X_pred = pd.DataFrame({
            "Pclass": [int(Pclass)],
            "Sex": [Sex],
            "Age": [int(Age)],
            "Fare": [float(Fare)]
        })
    print(X_pred)

    y_pred = model.best_estimator_.predict(X_pred).tolist()
    y_pred_proba = model.best_estimator_.predict_proba(X_pred).tolist()
    print(y_pred_proba)
    return {"Survived probability": y_pred_proba[0][1],
            "Predicted": y_pred[0]}

