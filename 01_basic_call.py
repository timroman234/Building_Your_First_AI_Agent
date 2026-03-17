"""
Step 2: A basic OpenAI API call.
This is the simplest possible interaction — send a message, get a response.
We'll build on this foundation in every step that follows.
"""
from openai import OpenAI
from dotenv import load_dotenv

# Load the API key from .env into the environment
load_dotenv()

# Create the OpenAI client
# It automatically reads OPENAI_API_KEY from the environment
client = OpenAI()

# Send a message and get a response
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful weather assistant."
        },
        {
            "role": "user",
            "content": "What can you help me with?"
        }
    ]
)

# Print the assistant's reply
print(response.choices[0].message.content)
