"""
Lesson: Your First Responses API Call
=====================================
This is the simplest possible call to the OpenAI Responses API.
No messages array, no roles – just pass a string and get a response.

Before running:
  pip install openai

"""
import os
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

# Create a client – it reads OPENAI_API_KEY from your environment automatically
client = OpenAI()

# -----------------------------------------------------------
# The simplest Responses API call
# -----------------------------------------------------------
response = client.responses.create(
    model="gpt-5.5",                        # Which model to use
    input="Explain what an AI agent is in one paragraph.",  # Just a string!
)

# .output_text is a handy shortcut to get the text response
print("=" * 60)
print("RESPONSE:")
print("=" * 60)
print(response.output_text)
print()