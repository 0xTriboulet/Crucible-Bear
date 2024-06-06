import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LinearRegression

# Constants
CHALLENGE = "bear2"
CRUCIBLE_URL = "https://crucible.dreadnode.io"
CHALLENGE_URL = "https://bear2.crucible.dreadnode.io"
CRUCIBLE_API_KEY = "hVrw7jvlh6wtwTwAPFmNa24iOAqd755O"

# Function to query the API
def query(input_data):
    headers = {"Authorization": CRUCIBLE_API_KEY}
    response = requests.post(f"{CHALLENGE_URL}/score", headers=headers, json=input_data)
    return response.json()

# Step 1: Explore the Endpoint
print(query({"hunger": 5}))
print(query({"hunger": 0}))
print(query({"hunger": -5}))
print(query({"hunger": 999}))
print(query({"hunger": 'test'}))
print(query({"hunger": 50.00000000000000000000001}))
print(query({"coef": 0.1}))

# Step 2: Data Collection
results = {}  # Store the results from the endpoint here
for h in range(101):
    response = query({"hunger": h})
    if 'outputs' in response:
        results[h] = response['outputs'][0]
    else:
        print(f"Invalid response for hunger {h}: {response}")

# Step 3: Analysis
df = pd.DataFrame(list(results.items()), columns=['Hunger', 'Happiness'])

# Plotting Hunger vs. Happiness
plt.figure(figsize=(10, 6))
plt.scatter(df['Hunger'], df['Happiness'], alpha=0.6)
plt.title('Hunger vs. Happiness Analysis')
plt.xlabel('Hunger Level')
plt.ylabel('Happiness Score')
plt.grid(True)
plt.show()

# Step 4: Model Replication
# Create a simple linear regression model
lr = LinearRegression()
lr.fit(df[['Hunger']], df['Happiness'])

# Model Validation
predicted_happiness = lr.predict(df[['Hunger']])
plt.figure(figsize=(10, 6))
plt.scatter(df['Hunger'], df['Happiness'], alpha=0.6, label='Actual')
plt.plot(df['Hunger'], predicted_happiness, color='red', label='Linear Fit')
plt.title('Hunger vs. Happiness with Linear Regression Fit')
plt.xlabel('Hunger Level')
plt.ylabel('Happiness Score')
plt.legend()
plt.grid(True)
plt.show()

# Step 5: Extraction and Reporting
coefficient = -.5 # lr.coef_[0]
print("Extracted Coefficient:", coefficient)

# Submit and receive flag
coef_data = {"coef": str(coefficient)}
response = query(coef_data)
print("Response:", response)

# Check if the response contains the flag and submit it
if 'flag' in response:
    def submit_flag(flag):
        url = f"{CRUCIBLE_URL}/api/submit-flag"
        headers = {"Authorization": CRUCIBLE_API_KEY}
        payload = {"challenge": CHALLENGE, "flag": flag}
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            if response.json().get("correct") is True:
                print("The flag was correct. Congrats!")
            else:
                print("The flag was incorrect. Keep trying!")
        else:
            print("There was an error submitting your flag")
            print(response.text)
    
    submit_flag(response['flag'])
else:
    print("Flag not found in the response. Please check the coefficient and try again.")

# Conclusion
print("Engaging in model extraction challenges offers a unique lens through which to understand and interact with machine learning models. This journey not only sharpens your analytical and problem-solving skills but also deepens your appreciation for the complexity inherent in these models. Through practical experience, you gain insights into how models interpret data and make predictions, equipping you with a versatile toolkit for tackling real-world data science challenges. Such endeavors underscore the importance of critical thinking and adaptability, foundational qualities for success in the ever-evolving landscape of machine learning and data science.")
