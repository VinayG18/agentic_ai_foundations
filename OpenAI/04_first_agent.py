"""
Lesson: Your First Agent with the Agents SDK
============================================
The Agents SDK makes it incredibly easy to build agents.
Just define an Agent with a model, name, and instructions, then run it.

Before running:
  pip install openai-agents
  export OPENAI_API_KEY="sk-..."

For Groq:
    pip install openai-agents openai python-dotenv
"""

import os # for Groq
from dotenv import load_dotenv
from openai import AsyncOpenAI # for Groq
from agents import Agent, Runner, OpenAIChatCompletionsModel # OpenAIChatCompletionsModel - Add when you use groq

load_dotenv()


# -----------------------------------------------------------
# Groq client
# -----------------------------------------------------------
groq_client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# -----------------------------------------------------------
# Groq model
# -----------------------------------------------------------
groq_model = OpenAIChatCompletionsModel(
    model="qwen/qwen3.6-27b",
    openai_client=groq_client,
)

# -----------------------------------------------------------
# Step 1: Define an Agent
# -----------------------------------------------------------
# An Agent needs:
#   - name: a label for identification and tracing
#   - instructions: the system prompt that defines behavior
#   - model: LLM
agent = Agent(
    name="History Tutor",
    instructions="""You are a friendly history tutor.
You answer history questions clearly and concisely.
Always include an interesting fun fact in your answers.""",
    # model="gpt-5.5",
    model=groq_model,
)

# -----------------------------------------------------------
# Step 2: Run the Agent
# -----------------------------------------------------------
# Runner.run_sync() is the synchronous way to execute an agent.
# (There's also an async version: await Runner.run())
print("--- Question 1 ---")
result = Runner.run_sync(agent, "Who was the first president of the United States?")
print(result.final_output)
print()

# -----------------------------------------------------------
# Run it again with a different question
# -----------------------------------------------------------
print("--- Question 2 ---")
result2 = Runner.run_sync(agent, "What caused World War I?")
print(result2.final_output)
print()


print("✅ You just built and ran your first agents!")
# print("   Check your traces at: https://platform.openai.com -> Dashboard -> Traces")