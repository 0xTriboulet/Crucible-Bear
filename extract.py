import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# Load the dataset
df = pd.read_csv('bear.csv')
print(df.head())

# Understanding the Data
# Display data types and check for missing values
print(df.info())

# Distribution of Numeric Data
# Plotting the distribution of 'tune'
df['tune'].plot(kind='hist', bins=25, title='Distribution of `tune` Feature')
plt.xlabel('Tune Value')
plt.ylabel('Frequency')
plt.show()

# Analyzing Categorical Data
# Plotting the frequency of bear types
df['bear'].value_counts().plot(kind='barh', title='Frequency of Bear Types')
plt.xlabel('Number of Occurrences')
plt.ylabel('Bear Type')
plt.show()

# Exploring Text Data
# Displaying unique characters in the 'val' column
unique_values = df['val'].unique()
print("Unique characters in the 'val' column:", unique_values)

# Step 3: Sorting and Filtering the Data
# Understanding Groupby Aggregations on a Pandas DataFrame
# Group by the bear type and aggregate to the average `tune` value
mean_tunes = df.groupby('bear')['tune'].mean()
print(mean_tunes)

# Sorting the Pandas DataFrame
# Sorting the DataFrame by 'tune' in descending order to see the top values
top_tunes = df.sort_values('tune', ascending=False).head(5)
print(top_tunes)

# Filtering Data for Specific Conditions
# Filtering to find entries where 'tune' values are above a certain threshold
high_tune_bears = df[df['tune'] > 90]
print(high_tune_bears.head(5))

# Applying Multiple Filters
# Applying multiple conditions to find a specific subset of data
specific_bears = df[(df['tune'] > 50) & (df['bear'] == 'Kodiak')]
print(specific_bears.head(5))

# Step 4: Solving the Challenge
# Analyzing and Identifying Key Data
# Sorting the data by 'tune' in descending order to focus on the highest values
sorted_data = df.sort_values('tune', ascending=False)
print(sorted_data.head())

# Extracting Critical Information
# Grouping by 'bear' type and taking the top entry for each group based on 'tune' value
top_bears = df.sort_values('tune', ascending=False).groupby('bear').head(1)
top_bears = top_bears.sort_values('bear')  # Sorting by bear type for easier visualization
print(top_bears)

# Decoding the Flag
# Extracting characters from the 'val' column
characters = top_bears['val'].tolist()
secret_word = ''.join(characters)
print("Secret Word:", secret_word)

# Removing unwanted characters
secret_word = [c for c in secret_word if c not in ['{', '}', ' ']]
secret_word = ''.join(secret_word)
print("Formatted Secret Word:", secret_word)

# Submitting Our Findings to the Endpoint
# Define your challenge URL and API key
CHALLENGE = "bear1"
CRUCIBLE_URL = "https://crucible.dreadnode.io"
CHALLENGE_URL = "https://bear1.crucible.dreadnode.io"
ARTIFACT_FILES = ['bear.csv']
CRUCIBLE_API_KEY = "hVrw7jvlh6wtwTwAPFmNa24iOAqd755O"

def query(input_data):
    response = requests.post(
        f"{CHALLENGE_URL}/score",
        headers={"Authorization": CRUCIBLE_API_KEY},
        json={"data": input_data}
    )
    return response.json()

response = query(secret_word)
print(response)


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
