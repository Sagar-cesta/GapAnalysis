"""
STEP 1 - Hello LiteLLM
----------------------
Goal: Verify that LiteLLM is installed correctly and can talk to OpenAI.

What this script does:
  1. Loads your OpenAI API key from the .env file
  2. Sends a simple chat message to GPT-4o-mini using LiteLLM
  3. Prints the response

Run this with:
  python step1_hello_litellm.py
"""

import os
from dotenv import load_dotenv
import litellm

# Load environment variables from .env file
load_dotenv()

# Read the API key (LiteLLM picks it up automatically from the environment)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key == "your-openai-api-key-here":
    print("ERROR: Please open the .env file and paste your real OpenAI API key.")
    exit(1)

print("Sending a message to GPT-4o-mini via LiteLLM...")
print("-" * 50)

# This is the simplest possible LiteLLM call
# "messages" is a list of chat turns, just like the OpenAI chat API
response = litellm.completion(
    model="gpt-4o-mini",        # Which LLM to use
    messages=[
        {
            "role": "user",
            "content": "Say hello and tell me what LiteLLM is in one sentence."
        }
    ]
)

# Extract and print the text reply
reply = response.choices[0].message.content
print("LLM Response:")
print(reply)
print("-" * 50)
print("SUCCESS: LiteLLM is working correctly!")
