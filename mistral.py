import os
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()


client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

response = client.beta.conversations.start(
    agent_id="ag_019ac49c2b2477dc926940b9d51e50f4",
    inputs="how do i make a simple agent from langchain in latest langchain version v-1.1.0",
)

print(response.outputs[0].content)