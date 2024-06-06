import requests
import base64
from PIL import Image
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import timm
from torchvision import transforms

# Constants
CHALLENGE = "bear3"
CRUCIBLE_URL = "https://crucible.dreadnode.io"
CHALLENGE_URL = "https://bear3.crucible.dreadnode.io"
CRUCIBLE_API_KEY = "hVrw7jvlh6wtwTwAPFmNa24iOAqd755O"
ARTIFACT_FILES = ['brown_bear.png', 'ping_pong.png']

# Download the Data
for artifact in ARTIFACT_FILES:
    url = f"{CRUCIBLE_URL}/api/artifacts/{CHALLENGE}/{artifact}"
    headers = {"Authorization": CRUCIBLE_API_KEY}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        with open(artifact, "wb") as file:
            file.write(response.content)
        print(f"{artifact} was successfully downloaded")
    else:
        print(f"Failed to download {artifact}")

# Function to encode image to base64
def get_encoded_image(image_filename):
    with open(image_filename, "rb") as f:
        image = base64.b64encode(f.read())
    return image

# Function to query the API
def query(input_data):
    response = requests.post(
        f"{CHALLENGE_URL}/score",
        headers={"Authorization": CRUCIBLE_API_KEY},
        json={"data": input_data}
    )
    return response.json()

# Displaying Images
brown_bear_image = Image.open("brown_bear.png")
ping_pong_image = Image.open("ping_pong.png")
brown_bear_image.show()
ping_pong_image.show()

# Testing the API endpoint with the provided images
image_data = get_encoded_image("brown_bear.png")
response = query(image_data.decode())
print("Response for 'brown_bear.png':", response)

image_data = get_encoded_image("ping_pong.png")
response = query(image_data.decode())
print("Response for 'ping_pong.png':", response)

# Fingerprinting Approach
# Initialize tracking DataFrame
tracking = pd.DataFrame()

# Setup plot for visualizing responses
fig, axs = plt.subplots(2, 5, figsize=(10, 5))
axs = axs.flatten()

# Load and display the base image
bear_image = Image.open("brown_bear.png")
for i, angle in enumerate(range(0, 360, 36)):
    # Rotate image and encode it to base64
    rotated_image = bear_image.rotate(angle)
    buffered = io.BytesIO()
    rotated_image.save(buffered, format="PNG")
    rotated_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Query the model endpoint with the rotated image
    response = query(rotated_image_base64)
    print(f"Image rotated at angle {angle}°, score: {response}")

    # Store response in DataFrame
    tracking.loc[i, "base_image"] = "brown_bear.png"
    tracking.loc[i, "rotation"] = angle
    tracking.loc[i, "brown_bear_score"] = response["brown bear"]

    # Display the rotated image and score
    axs[i].imshow(rotated_image)
    axs[i].set_title(f'Score: {response["brown bear"]:0.4f}')
    axs[i].axis("off")
plt.show()

# Example output logs
print(tracking.head())

# Pretrained Image Models
# Function to load an image and transform it for model inference
def load_and_transform_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to perform prediction using a specified model
def predict_image(model_name, image_tensor):
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
    return logits.argmax(dim=1), logits

# Testing Pretrained Models
image_tensor = load_and_transform_image("brown_bear.png")
model_names = ["mobilenetv3_large_100", "efficientnet_b0", "resnet18"]
BROWN_BEAR_INDEX = 294  # Index for brown bear in ImageNet

# Test each model and print out the probability of 'brown bear'
for model_name in model_names:
    prediction, logits = predict_image(model_name, image_tensor)
    probs = torch.softmax(logits, dim=1)  # Convert logits to probabilities
    print(f"Prediction from {model_name}: {prediction}")
    print(f"Brown bear probability: {probs[0][BROWN_BEAR_INDEX]:.4f}")

# Fingerprinting offline models
# Re-using the rotated images to test each offline model
for i, angle in enumerate(range(0, 360, 36)):
    rotated_image = bear_image.rotate(angle)  # Rotate image
    rotated_image.save("temp.png")  # Save as a temporary file
    image_tensor = load_and_transform_image("temp.png")  # Transform image for model inference

    for model_name in model_names:
        prediction, logits = predict_image(model_name, image_tensor)
        probs = torch.softmax(logits, dim=1)
        # Store each model's probability for the brown bear class in tracking DataFrame
        tracking.loc[i, f"{model_name}_score"] = probs[0][BROWN_BEAR_INDEX].item()

# Plotting the results for visual comparison
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
for model_name, color in zip(model_names, ["blue", "red", "orange"]):
    tracking.plot(
        x="rotation",
        y=f"{model_name}_score",
        style="-.",
        label=f"{model_name} predictions",
        ax=axs[0],
        color=color
    )

tracking.plot(
    x="rotation",
    y="brown_bear_score",
    style="-",
    label="API predictions",
    ax=axs[1],
    color="black"
)

axs[0].set_title("Model Predictions by Rotation")
axs[1].set_title("API Predictions by Rotation")
axs[0].set_ylabel("Probability")
axs[1].set_ylabel("Probability")
plt.legend()
plt.show()

# Analyzing Correlations
# Calculate and print the correlation of each model's scores with the API scores
for model_name in model_names:
    correlation = np.corrcoef(tracking['brown_bear_score'], tracking[f'{model_name}_score'])[0, 1]
    print(f"Correlation of {model_name} with API: {correlation:.4f}")

# Getting the Flag
# Load and transform the ping_pong.png image
image_tensor = load_and_transform_image("ping_pong.png")

# Make predictions using the identified model
prediction, logits = predict_image('resnet18', image_tensor)
probs = torch.softmax(logits, dim=1)

PING_PONG_IDX = 722  # Index for ping pong ball in ImageNet
ping_pong_prediction = probs[0][PING_PONG_IDX].item()
print("Predicted resnet18 probability for 'ping pong ball':", ping_pong_prediction)

# Function to send the predicted probability to the API and get the flag
def get_flag(pingpong_prob):
    response = requests.post(
        f"{CHALLENGE_URL}/score",
        headers={"Authorization": CRUCIBLE_API_KEY},
        json={"pingpong": pingpong_prob}
    )
    return response.json()

# Fetch the flag using the predicted probability
flag_response = get_flag(ping_pong_prediction)
print("Flag Response:", flag_response)

# Submitting the flag
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

# If the flag was found, submit it
if 'flag' in flag_response:
    submit_flag(flag_response['flag'])
else:
    print("Flag not found in the response. Please check the prediction and try again.")
