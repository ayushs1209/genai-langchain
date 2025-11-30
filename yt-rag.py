from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=H6Qd3Joo7Ac")

result = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

split_text = splitter.split_text(result[0].page_content)

# print(result[0].page_content)
# print(split_text)

# for text in split_text :
#     print(text + " \n \n")

template = PromptTemplate(
    template= """
    You are a Helpful AI assistant whose job is to summarize the content of the yt-video based on the {text},
    """,
    input_variables=['text']
)

parser = StrOutputParser()

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash"
)

chain = template | model | parser

output = chain.invoke({"text" : result[0].page_content})

print(output)
