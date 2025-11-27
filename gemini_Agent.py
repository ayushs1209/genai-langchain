import os
from typing import List, Dict, Any

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# 1. Define tools the agent can call
@tool
def get_weather(city: str) -> str:
    """Return a fake weather report for a given city."""
    return f"It's always sunny in {city}!"

tools = [get_weather]






# 2. Configure Gemini chat model via langchain-google-genai
#    You can swap model name to whatever Gemini variant you have access to:
#    e.g. "gemini-2.0-flash", "gemini-1.5-pro", etc.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.3,
    max_tokens = 2500
)

# 3. Create the agent using LangChain v1.x create_agent API
agent = create_agent(
    model=llm,                       # can be a ChatModel instance
    tools=tools,                     # list of @tool functions
    system_prompt=(
        "You are a helpful assistant. "
        "Use tools when needed, otherwise answer directly."
    ),
)

# 4. Invoke the agenthe messages list in OpenAI-style format
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Why is the sky blue in color?",
            }
        ]
    }
)

    # Result is a dict with a 'messages' list; last one is the agent's reply
messages = result["messages"][-1].content
print("AGENT:", messages)

