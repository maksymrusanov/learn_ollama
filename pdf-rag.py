from langchain_community.document_loaders import OnlinePDFLoader
from langchain_unstructured import UnstructuredLoader

doc_path = "/home/max/Documents/ollama/data/BOI.pdf"
model = "llama3.2"
if doc_path:
    loader = UnstructuredLoader(doc_path)
    data = loader.load()
    print("done loading")
else:
    print("upload PDF file")
content = data[0].page_content


from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
chunks = text_splitter.split_documents(data)
print("done spliting")

import ollama

ollama.pull("nomic-embed-text")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="simple-rag",
)
print("done adding to vector database ")
