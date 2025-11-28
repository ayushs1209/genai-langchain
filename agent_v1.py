from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.tools import tool
import os

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

@tool
def run_command(command) :
    """
    this is a python function which is used to perform a command in the system terminal
    you can use this tool to make files and directories on the users command

    """
    os.system(command=command)

tools = [run_command]

agent = create_agent(model=model,tools=tools, system_prompt="You are a helpful coding ai assistant. who has the capabilities to create new files and new folders and append code into them to help the user. You have the ability to create files and folders. you can create a file using the touch command example : if the user wants a file with filename laptop.py you can just run the command touch from the tool run_command, you can do the same with the the create directory, you can use the mkdir command to make a file directory  and use the tool run_command to actually run the command")


while (1) :

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": input("> ")
                }
            ]
        }
    )
    messages = result["messages"][-1].content
    print("AGENT:", messages)

