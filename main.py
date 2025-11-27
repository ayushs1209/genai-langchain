# pip install -qU "langchain[anthropic]" to call the model

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI


from dotenv import load_dotenv
load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

SYSTEM_PROMPT = """
You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- say_my_name: use this to return the user the string Hiesenberg as from the tool output

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they are asking their name, you use the say_my_name tool to send back the string."""

def say_my_name() :
    """ You should return Heisenberg whenever you're asked what's my name
    """
    return "Heisenberg"

agent = create_agent(
    model=model,
    tools=[get_weather, say_my_name],
    system_prompt=SYSTEM_PROMPT,
)

# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the capital of india"}]}

)


print(result["messages"][-1].content[0]["text"])


# for msg in result:
#     if isinstance(msg, AIMessage):
#         print(msg.content)