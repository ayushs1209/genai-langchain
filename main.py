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

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the capital of india"}]}

)


print(result["messages"][-1].content[0]["text"])


# for msg in result:
#     if isinstance(msg, AIMessage):
#         print(msg.content)