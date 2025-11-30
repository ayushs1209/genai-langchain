from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

loader = PyPDFLoader(
    file_path = "attention-is-all-you-need.pdf"
)

docs = loader.load()

print(docs[0].metadata)



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100,
)

