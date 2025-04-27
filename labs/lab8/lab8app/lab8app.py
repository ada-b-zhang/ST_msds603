from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

# Load model
model_path = "mlruns/1/2e99afaa42f34728923da91b2988af07/artifacts/metaflow_train"
model = mlflow.pyfunc.load_model(model_path)

# Initialize FastAPI app
app = FastAPI()

# Define the class mapping (numeric label to flower name)
class_mapping = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

# Define the request body format
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define prediction endpoint
@app.post("/predict")
def predict(features: IrisFeatures):
    # Convert request data to DataFrame
    input_df = pd.DataFrame([features.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)
    predicted_class = prediction[0]  # First (and only) prediction

    # Map numeric class to label
    class_name = class_mapping.get(predicted_class, "unknown")

    return {
        "prediction_numeric": int(predicted_class),
        "prediction_label": class_name
    }
