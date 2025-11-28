from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash"
)

template = PromptTemplate(
    template="Tell me in detail about {topic} \n",
    input_variables=['topic']
)

parser = StrOutputParser()

template2 = PromptTemplate(
    template=" summarize the following in 5 points {text} \n",
    input_variables=['text']
)

report_chain = template | model | parser

chain = RunnableBranch(
    (lambda x : len(x.split()) > 5000 , template2 | model | parser),
    (RunnablePassthrough())
)

final_chain = report_chain | chain 

print(final_chain.invoke({'topic' : 'black hole'}))

