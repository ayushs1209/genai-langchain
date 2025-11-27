import os
from typing import List, Dict, Any

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Return a fake weather report for a given city."""
    return f"It's always sunny in {city}!"

@tool
def say_my_name() :
    """ You should return Heisenberg whenever you're asked what's my name
    """
    return "Heisenberg"

tools = [get_weather,say_my_name]



llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.3,
    max_tokens = 2500
)

agent = create_agent(
    model=llm,                       # can be a ChatModel instance
    tools=tools,                     # list of @tool functions
    system_prompt=(
        "You are a helpful assistant. "
        "Use tools when needed, otherwise answer directly."
    ),
)


user_input = input("> ")

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": user_input,
            }
        ]
    }
)



messages = result["messages"][-1].content
print("AGENT:", messages)

