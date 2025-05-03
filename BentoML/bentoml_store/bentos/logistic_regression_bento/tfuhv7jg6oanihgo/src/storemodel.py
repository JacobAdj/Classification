import bentoml
import joblib

# Path to your Joblib-saved model
joblib_model_path = "./pretrainedmodels/logistic_regression_model3.pkl"

# Load the model from Joblib file
model = joblib.load(joblib_model_path)

# Save the model in the BentoML model store
saved_model = bentoml.sklearn.save_model("iris3", model)

print(f"Model saved: {saved_model}")
