from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

#Load dotend
load_dotenv()


#firstly load up the document
loader = PyPDFLoader("KB.pdf")
kb = loader.load()



#Split the document into chunks
textSplitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
kb_chunks = textSplitter.split_documents(kb)


#Now let's embed the chunks. (Embedings = Vectors that represent each chunk/word)
#Major difference between a token and an embedding is it's relationship with other words
#Vectors being multidimentional figures relationships between words
#Tokens are like IDs given to each word
#Embeddings help predict the words around a token. (ONLY MAPS RELATIONS)


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

db = Qdrant.from_documents(
    kb_chunks,
    embeddings,
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    collection_name="campus_rag"
)

