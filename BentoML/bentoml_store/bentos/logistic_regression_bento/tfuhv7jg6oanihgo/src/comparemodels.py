import bentoml
import joblib

# Load two models from BentoML model store
model1 = bentoml.sklearn.load_model("iris3")
model2 = bentoml.sklearn.load_model("iris10")
model3 = bentoml.sklearn.load_model("iris1000")

print("Models loaded successfully!")


from sklearn.metrics import accuracy_score

def compare_models(model_1, model_2, model_3, X_test, y_test):
    # Predict with all three models
    y_pred_1 = model_1.predict(X_test)
    y_pred_2 = model_2.predict(X_test)
    y_pred_3 = model_3.predict(X_test)

    # Compute accuracy scores
    acc_1 = accuracy_score(y_test, y_pred_1)
    acc_2 = accuracy_score(y_test, y_pred_2)
    acc_3 = accuracy_score(y_test, y_pred_3)

    print(f"Model 1 Accuracy: {acc_1:.4f}")
    print(f"Model 2 Accuracy: {acc_2:.4f}")
    print(f"Model 3 Accuracy: {acc_3:.4f}")

    return {
        "Model_1_Accuracy": acc_1, 
        "Model_2_Accuracy": acc_2, 
        "Model_3_Accuracy": acc_3
    }


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target  # Features and labels

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print sample test data
print("Sample Test Features:\n", X_test[:5])  # Show first 5 test examples
print("Sample Test Labels:\n", y_test[:5])  # Show first 5 labels

print('n labels' , len(y_test))

# Randomly change values from 1 to 2 in y_test
num_changes = 5  # Number of random replacements
indices_to_change = np.random.choice(np.where(y_test == 1)[0], size=num_changes, replace=False)
y_test[indices_to_change] = 2  # Replace 1 with 2

# Print sample modified test labels
print("Modified y_test:\n", y_test[:10])  # Show first 10 labels

# Compare models
result = compare_models(model1, model2, model3, X_test, y_test)
print(result)
