import requests

# URL of the server
url = "http://127.0.0.1:3000/predict"

# Data to send in the request body
import json

# Define the data
iris_data = {
    "input_data": {  
        "sepal length (cm)": 7.7, 
        "sepal width (cm)": 22.6, 
        "petal length (cm)": 6.9, 
        "petal width (cm)": 2.3
    }
}


# Save to a JSON file
# with open("iris_data.json", "w") as json_file:
#     json.dump(iris_data, json_file, indent=4)

#print("JSON file created successfully!")


# Sending the POST request
headers = {"Content-Type": "application/json"}
response = requests.post(url, json=iris_data, headers=headers)

# Checking the response
print(f"Status Code: {response.status_code}")
print(f"Response Data: {response.json()}")
