"""
===================================================================
OCI Enterprise AI Agents - Your First Agent (OCI Responses API)
===================================================================
Companion code for the "OCI Enterprise AI Agents - Beginner Course"

Same 4 math tools and test cases as the LangChain demo - but here the
OCI Responses API IS the agentic loop. We build the loop by hand once,
so you can SEE what frameworks like LangChain do for you behind the scenes.

Prerequisites:
  pip install openai python-dotenv

Authentication (one-time setup):
  1. OCI Console -> Generative AI -> API Keys -> create a key
  2. Add the IAM policy that lets the key use the GenAI service
  3. Put the key in a .env file next to this script:
       OCI_GENAI_API_KEY=your-key-here
  4. Set OCI_REGION and OCI_PROJECT_ID below to match your tenancy

Reference:
  https://docs.oracle.com/en-us/iaas/Content/generative-ai/agents.htm
===================================================================
"""

import os
import json
import math
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# -- Load the API key ----------------------------------------------------
#   "We keep the key out of the code in a .env file. This line finds that
#    file sitting right next to the script, no matter where we run python
#    from." Path(__file__) is this file; .parent is its folder.
script_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=script_dir / ".env")

# Confirm the key loaded WITHOUT printing the key itself (never print secrets).
print("OCI_GENAI_API_KEY found:", bool(os.getenv("OCI_GENAI_API_KEY")))

# -- Configuration -------------------------------------------------------
#   "Three things make this point at OCI instead of OpenAI: the region in
#    the URL, the model name, and your project OCID. Change these to yours."

OCI_REGION = "us-chicago-1"        # <-- CHANGE to your region

OCI_BASE_URL = (
    f"https://inference.generativeai.{OCI_REGION}"
    f".oci.oraclecloud.com/openai/v1"
)

MODEL = "openai.gpt-oss-120b"       # strong reasoning model on OCI

# OCI Console -> Generative AI -> Projects -> your project -> copy OCID
OCI_PROJECT_ID = "ocid1.generativeaiproject.oc1.us-chicago-1.amaaaaaanf25m3qagww3n7gfydnnjoyidrbimiuribdfcq5t6quonulugzea"

# -- Create the Client ---------------------------------------------------
#   "Here's the trick - it's the standard OpenAI client, but we swap in
#    OCI's base_url and pass our project header. Same SDK, different backend."
def create_client():
    api_key = os.getenv("OCI_GENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OCI_GENAI_API_KEY in your .env file.")
        exit(1)

    return OpenAI(
        api_key=api_key,
        base_url=OCI_BASE_URL,
        default_headers={"OpenAI-Project": OCI_PROJECT_ID},
    )

# =======================================================================
# STEP 1: Define the Math Tools
# =======================================================================
#   "An agent is only as useful as its tools. These are 4 plain Python
#    functions - the actual work happens here."

def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

def divide(a: float, b: float) -> str:
    """Divide the first number by the second."""
    if b == 0:
        return "Error: Cannot divide by zero"  # graceful, not a crash
    return str(a / b)

def square_root(number: float) -> str:
    """Calculate the square root of a number."""
    if number < 0:
        return "Error: Cannot take square root of a negative number"
    return str(math.sqrt(number))

#   "The model never SEES this Python at runtime - it only knows the tools
#    we declare to it. So we describe each one as JSON: its name, what it
#    does, and its inputs. That JSON is the contract the model uses to
#    request a call, and the API uses to check the arguments. In LangChain
#    the @tool decorator generated this JSON for you - here we write it out."

MATH_TOOLS = [
    {
        "type": "function",
        "name": "add",
        "description": "Add two numbers together. Use for addition operations.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "type": "function",
        "name": "multiply",
        "description": "Multiply two numbers together. Use for multiplication.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "type": "function",
        "name": "divide",
        "description": "Divide the first number by the second. Returns error if dividing by zero.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "type": "function",
        "name": "square_root",
        "description": "Calculate the square root of a number.",
        "parameters": {
            "type": "object",
            "properties": {
                "number": {"type": "number", "description": "The number to take square root of"},
            },
            "required": ["number"],
        },
    },
]

#   "One small bridge: the model returns a tool NAME as text. This map
#    turns that name back into the real Python function to call."
TOOL_DISPATCH = {
    "add": add,
    "multiply": multiply,
    "divide": divide,
    "square_root": square_root,
}

# =======================================================================
# STEP 2: The Agent Loop <- the heart of the demo
# =======================================================================
#   "This is what an 'agent' really is. The model doesn't do math - it
#    decides WHICH tool to call. We run the tool, hand the answer back, and
#    let it decide again. We repeat until it stops asking for tools.
#    LangChain hides this loop. Today we build it ourselves."

def run_agent(client, question: str):
    print(f"👤 User: {question}")
    print("-" * 50)

    # First call: send the question + the tools the model is allowed to use.
    response = client.responses.create(
        model=MODEL,
        input=question,
        tools=MATH_TOOLS,
    )

    iteration = 0
    while True:
        iteration += 1

        # Did the model ask to call any tools this round?
        tool_calls = [item for item in response.output
                      if item.type == "function_call"]

        # No tool calls means the model is finished - print its final answer.
        if not tool_calls:
            print(f"🤖 Agent: {response.output_text}")
            break

        #   "The model asked for a tool. We run it for real, right here."
        tool_results = []
        for tc in tool_calls:
            func = TOOL_DISPATCH.get(tc.name)
            args = json.loads(tc.arguments)
            result = func(**args)
            print(f"  🔧 Step {iteration}: {tc.name}({args}) -> {result}")

            # Package the result so the model can read it on the next round.
            tool_results.append({
                "type": "function_call_output",
                "call_id": tc.call_id,
                "output": str(result),
            })

        #   "Feed the results back. previous_response_id lets OCI remember
        #    the conversation so far - no need to resend everything."
        response = client.responses.create(
            model=MODEL,
            previous_response_id=response.id,
            input=tool_results,
        )

    print("=" * 50)
    print()

# =======================================================================
# EXAMPLE 1: Simple Chat (no tools)
# =======================================================================
#   "Before tools, the simplest possible call: text in, text out. Proves
#    our connection to OCI works."

def simple_chat_example(client):
    print("=" * 60)
    print("EXAMPLE 1: Simple Chat (no tools)")
    print("=" * 60)

    response = client.responses.create(
        model=MODEL,
        input="Explain what an AI agent is in 3 sentences, "
              "using a real-world analogy.",
    )
    print(f"\n🤖 {response.output_text}\n")

# =======================================================================
# EXAMPLE 2: Math Agent - 4 test cases, easy -> hard
# =======================================================================

def math_agent_example(client):
    print("=" * 60)
    print("EXAMPLE 2: Math Agent (add, multiply, divide, square_root)")
    print("=" * 60)
    print("\n=== Available Tools ===")
    print("• add: Add two numbers together. Use for addition operations.")
    print("• multiply: Multiply two numbers together. Use for multiplication.")
    print("• divide: Divide the first number by the second. Returns error if dividing by zero.")
    print("• square_root: Calculate the square root of a number.\n")

    run_agent(client, "What is 42 + 58?")
    run_agent(client, "What is 15 multiplied by 8, then divided by 3?")


def main():
    print("===================================================================")
    print("OCI Enterprise AI Agents – First Agent (Responses API)")
    print("===================================================================")
    
    client = create_client()
    print(f"Endpoint : {OCI_BASE_URL}")
    print(f"Model    : {MODEL}\n")

    simple_chat_example(client)
    math_agent_example(client)

    print("-" * 50)
    print("Done! Same agent as the LangChain demo – but we built the")
    print("loop ourselves, so now you know what the framework hides.")
    print("-" * 50)


if __name__ == "__main__":
    main()