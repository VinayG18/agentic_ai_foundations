"""
==================================================================
LangChain for AI Agents - Companion Code
==================================================================
Lesson 1: Core Building Blocks
    - Models, Prompts, Chains, Output, Parsers, Memory, Tools

Demo order (matches the lessons):
    - Lesson 1 -> Models, Prompts, Chains, Memory
    - Lesson 2 -> Tools

Prerequisites:
    pip install langchain langchain-openai python-dotenv

Setup:
    Create a .env file with: OPENAI_API_KEY=sk-your-key-here
==================================================================
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
from pathlib import Path
from dotenv import load_dotenv

# Find the .env file sitting next to THIS script (not whenever we ran from).
# Path(__file__) = this file; .parent = its folder. This makes the key load
# correctly no matter which directory you launch python from.
script_dir = Path(__file__).resolve().parent
env_path = script_dir / ".env"

# print("Current working directory: ", os.getcwd())
# print("Script directory:", script_dir)
# print("Using .env from:", env_path)

load_dotenv(dotenv_path=env_path)

# Quick sanity check that key actually loaded before we call the model.
# print("OPENAI_API_KEY found:", bool(os.getenv("OPENAI_API_KEY")))
print("NVIDIA_API_KEY found:", bool(os.getenv("NVIDIA_API_KEY"))) # NVIDIA KEY

# ==================================================================
# 1. MODELS - The Reasoning Engine
# ==================================================================
# The "model" is the LLM itself - the part that actually thinks.
# init_chat_model gives ONE interface to every provider: to switch from
# OpenAI to Anthropic or Google, you change only the string below.

# from langchain.chat_models import init_chat_model
from langchain_nvidia_ai_endpoints import ChatNVIDIA
# model = init_chat_model("openai:gpt-5.5")
# model = init_chat_model("anthropic:claude-3-5-sonnet-latest")
# model = init_chat_model("google_genai:gemini-2.0-flash")
model = ChatNVIDIA(model="openai/gpt-oss-120b", temperature=1) # NVIDIA Model
# openai/gpt-oss-120b - other options: messages=[{"role":"user","content":""}], temperature=1, top_p=1, max_tokens=4096, stream=False

# The simplest possible use: send text in, get an answer back.
# .invoke() is the universal "run it" method across LangChain.
# response = model.invoke("What is LangChain in one sentence?")
response = model.invoke(
    "Who won the 2026 Women's Premier League (WPL)? "
    "Answer with only the team name."
)
print("=== Model Response ===")
print(response.content) # .content = just the text of the reply
print()

# ==================================================================
# 2. PROMPT TEMPLATES - Steering the Model
# ==================================================================
# A prompt template is a reusable sentence with blanks ({placeholders})
# you fill in later - write the wording once, reuse it many times

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# PromptTemplate = a single plain-text string with blanks.
simple_template = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} to a complete beginner in 2-3 sentences."
)

# .format() fills the blank and returns the finished text.
formatted = simple_template.format(topic="AI agents")
print("=== Formatted Prompt ===")
print(formatted)
print()

# ChatPromptTemplate = built from ROLES (system/human), which is how chat
# models expect their input. "system" sets behavior, "human" is the user turn.
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding tutor. Keep answer short and clear"),
    ("human", "Explain {concept} with a simple Python example."),
])

# .format_messages() fills the blanks and returns a LIST of messages.
messages = chat_template.format_messages(concept="list comprehension")
# {concept} in the template is the empty blank, and concept="list comprehension" is you handing it the word that goes in the blank.
print("=== Chat Messages ===")
for msg in messages:
    print(f" [{msg.type}]: {msg.content[:80]}...")
print()

# ==================================================================
# 3. CHAINS - Connecting the pieces with LCEL
# ==================================================================
# The pipe | glues components into a pipeline. Each step's output flows
# into the next, left to right: prompt | model | parser.

from langchain_core.output_parsers import StrOutputParser

# prompt fills the blank -> model answer -> parser pulls out clean text.
# StrOutputParser just extracts the plain string from the model's reply
# object, so you don't have to write .content yourself every time.
chain = chat_template | model | StrOutputParser()

# Run the whole pipeline with a single .invoke().
result = chain.invoke({"concept": "for loops"})
print("=== Chain Output ===")
print(result)
print()

# ==================================================================
# 4. MEMORY - Giving the Model Context
# ==================================================================
# Models are stateless - they forgot everything between calls.
# "Memory" is simply us storing past messages and feeding them back in.
# (Modern LangChain uses ChatMessageHistory; the old
# ConversationBufferMemory is deprecated and out of the core package.)

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# A simple in-memory store that holds the conversation
memory = InMemoryChatMessageHistory()

# Hand-build a short conversation so we have something to "remember"
memory.add_message(HumanMessage(content="My name is Vinay and I'm learning LangChain"))
memory.add_message(AIMessage(content="Nice to meet you, Vinay! LangChain is great choice"))
memory.add_message(HumanMessage(content="What tools should I learn first?"))
memory.add_message(AIMessage(content="Start with PromptTemplates and simple chains, then move to tools and agents"))

print("=== Memory Content ===")
for msg in memory.messages:
    print(f" [{msg.type}]: {msg.content[:80]}...")
print()

# The "placeholder" slot is where the sorted messages get injected into the
# prompt, so the model can SEE the earlier conversation.
chat_with_memory = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful tutor. Use the conversation history to personalize your responses."),
    ("placeholder", "{history}"), # past messages get dropped in here
    ("human", "{question}"),
])

chain_with_memory = chat_with_memory | model | StrOutputParser()

# We passed the stored history in alongside the new question.
# Watch the model correctly recall the name "Vinay" - that's "memory".
result = chain_with_memory.invoke({
    "history": memory.messages,
    "question": "What was my name again?"
})

print("=== Memory Aware Response ===")
print(result)
print()