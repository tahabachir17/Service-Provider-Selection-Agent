import os
from openai import OpenAI

# Initialize the client exactly as your pipeline will
client = OpenAI(
    api_key=os.environ.get("AIzaSyDmfm671jXA1UCJJXXy_UvNHGuFS9z65wo"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Fetch and print the available models
available_models = client.models.list()

print("Models available to your API Key:")
for model in available_models.data:
    if "pro" in model.id:
        print(f"✅ {model.id}")