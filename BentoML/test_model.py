import bentoml

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model3 = bentoml.sklearn.load_model("iris1000")

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target  # Features and labels

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred_3 = model3.predict(X_test)

acc_3 = accuracy_score(y_test, y_pred_3)

# Assert that accuracy is greater than 75%
assert acc_3 > 0.9, f"Assertion failed: Accuracy {acc_3:.2%} is below 90%"