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
from langchain_groq import ChatGroq

# Create a client – it reads OPENAI_API_KEY from your environment automatically
# client = OpenAI() ->OpenAI Responses API -> .responses.create()
# client = ChatGroq(model="qwen/qwen3.6-27b") # -> Groq + LangChain -> .invoke()
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# -----------------------------------------------------------
# The simplest Responses API call
# -----------------------------------------------------------
response = client.responses.create(
    model="qwen/qwen3.6-27b",                        # Which model to use
    input="Explain what an AI agent is in one paragraph.",  # Just a string!
)
# response = client.invoke(
#     "Explain what an AI agent is in one paragraph."
# )

# .output_text is a handy shortcut to get the text response
print("=" * 60)
print("RESPONSE:")
print("=" * 60)
print(response.output_text)
print()

# -----------------------------------------------------------
# You can also pass messages (like Chat Completions)
# -----------------------------------------------------------
response2 = client.responses.create(
    model="qwen/qwen3.6-27b",
    instructions="You are a helpful teacher who explains things simply.",  # System prompt
    input=[
        {"role": "user", "content": "What is the difference between an agent and a chatbot?"}
    ],
)

print("=" * 60)
print("RESPONSE WITH INSTRUCTIONS:")
print("=" * 60)
print(response2.output_text)