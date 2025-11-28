from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()


template1 = PromptTemplate(
    template="Give me a joke on {topic}",
    input_variables=['topic'],
)


template2 = PromptTemplate(
    template="Explain the following joke \n {joke}",
    input_variables=['joke']
)


template3 = PromptTemplate(
    template=" Rate this joke on a scale of 10 on funniness \n {funny}",
    input_variables=['funny']
)

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash"
)

parser = StrOutputParser()

joke_gen_chain = template1 | model | parser

parallel_chain = RunnableParallel({
    "joke" : RunnablePassthrough(),
    "explain" : template2 | model | parser,
    "rate" : template3 | model | parser
})

final_chain = joke_gen_chain | parallel_chain

result = final_chain.invoke({"topic" : "monkey"})

print("Joke : " +  result['joke'] + "\n \n ")
print("Explain : " + result['explain'] + "\n \n ")
print("Rating :  " + result['rate'] + "\n \n ")


final_chain.get_graph().print_ascii()