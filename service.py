# from __future__ import annotations
import bentoml
import pandas as pd


my_image = bentoml.images.Image(python_version="3.11") \
        .python_packages("torch", "transformers")

@bentoml.service(
    image=my_image,
    resources={"cpu": "2"},
    traffic={"timeout": 30},
)


class LogisticRegressionService:

    def __init__(self) -> None:

         # Load model into pipeline 
        self.model_name = "Iris"

        self.model = bentoml.sklearn.load_model("iris1000")

    
    @bentoml.api 
    def predict(self, input_data: dict) -> dict:
        """Handles prediction using the trained model."""
        # Convert JSON input into a dictionary
        print(input_data)

        # data_dict = input_data.dict()  # Extract dictionary
        features = pd.DataFrame([input_data])

        # Make prediction
        prediction = self.model.predict(features)

        return {"prediction": int(prediction[0])}
    

    @bentoml.api
    def health(self, input_data: dict) -> str:
    
        return input_data.input_data
