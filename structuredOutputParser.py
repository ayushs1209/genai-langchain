from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field 
from typing import Literal
from dotenv import load_dotenv

load_dotenv()


class Output(BaseModel):
    name :str =  Field(description= "The name of the person")
    age : int = Field(gt=10, description="The age of the person")
    address : str = Field(description="The address of the person")
    isMarried :  Literal['True','False']

parser = PydanticOutputParser(pydantic_object=Output)


model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 1.3
)

template = PromptTemplate(
    template= "Generate the name age address and isMarried for a {place} person \n {format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction" : parser.get_format_instructions()}

)

template2 = PromptTemplate(
    template= "Generate 6 random lines of personal data like the person does something, he likes this fruit, sport, show etc.  based on the given json data {details} \n",
    input_variables=['details']
)



# prompt = template.invoke({"place" : "nepalian"})

# result = model.invoke(prompt)

# final = parser.parse(result.content)

# prompt2 = template2.invoke({"details" : final})

# newResult = model.invoke(prompt2)

chain = template | model | parser | template2 | model

final = chain.invoke({"place" : "japanese"})

# print(final)


print(final.content)
# print("\n \n \n ")
# print(newResult.content)



