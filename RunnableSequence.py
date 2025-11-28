from langchain_core.runnables import RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


template = PromptTemplate(
    template=" Give me a joke on the topic {topic}",
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash"
)

parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke ({"topic" : "table"})

print(result)


