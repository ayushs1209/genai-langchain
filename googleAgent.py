from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
from dotenv import load_dotenv

load_dotenv()



@tool
def get_weather( city ) :
    """
    a helper tool function that mocks an api request and fetches the weather Data for the given city
    """

    # return f"its always sunny in {city}"

    response = requests.get(f"https://wttr.in/{city}?format=3")

    if response.status_code == 200:
        data = response.text
    else:
        data = "Something went wrong while fetching the weather details"

    return data


tools = [get_weather]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

agent = create_agent(
    model = llm,
    tools=tools,
    system_prompt="You are a helpful assistant. You start every conversation with Konnichiwa. You are provided with some tools for you to make your experience more realistic and interactive. the tools provided to you are get_weather. You should be able to use this tool to fetch the weather details of a city based on the user input."
)

while(1):
    result = agent.invoke(
    {
        "messages": [
                        {
                            "role": "user", "content": input("> ")
                        }
                    ]
    }
    )

    messages = result["messages"][-1].content
    print("AGENT:", messages) 


