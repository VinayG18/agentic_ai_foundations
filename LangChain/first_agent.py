#------------------------------------------------
# STEP 1: Initialize the Model
#------------------------------------------------

from langchain.chat_models import init_chat_model
model = init_chat_model("openai:gpt-4o-mini")

#------------------------------------------------
# STEP 2: Define Your Tools
#------------------------------------------------
# Each tool needs: a name, a clear docstring, and type hints. LLM reads these to decide WHEN and HOW to use each tool.

from langchain_core.tools import tool
import math

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together. Use for addition operations."""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together. Use for multiplication operations."""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divide the first number by the second. Returns error if dividing by zero."""
    if b == 0:
        return "Error: Cannot divide by zero"
    return a / b

@tool
def square_root(number: float) -> float:
    """Calculate the square root of a number."""
    if number < 0:
        return "Error: Cannot take square root of a negative number"
    return math.sqrt(number)

tools = [add, multiply, divide, square_root]

#------------------------------------------------
# STEP 3: Create the Agent
#------------------------------------------------
# create_agent build a full ReAct loop:
#   Reason -> Act (call tool) -> Observe -> Repeat

from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=tools,
)

# --------------------------------------------------
# STEP 4: Run the Agent!
# --------------------------------------------------

def run_agent(question: str):
    """Run the agent and print the execution trace."""
    print(f"\U0001F468 User: {question}")
    print("-" * 50)

    result = agent.invoke({
        "messages": [("user", question)]
    })
    print("\U0001F680 Agent:", result)

# Simple: single tool call
run_agent("What is 42 + 58?")

# Medium: multiple tool calls in sequence
run_agent("What is 15 multiplied by 8, then divided by 3?")

# Complex: the agent must plan a multi-step approach
run_agent(
    "I have a rectangle with width 12 and height 7. "
    "What is its area, and what is the square root of that area?"
)