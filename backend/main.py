from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


import os

#Load dotend
load_dotenv()


#firstly load up the document
loader = PyPDFLoader("KB.pdf")
kb = loader.load()



#Split the document into chunks
textSplitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
kb_chunks = textSplitter.split_documents(kb)


'''#Now let's embed the chunks. (Embedings = Vectors that represent each chunk/word)
#Major difference between a token and an embedding is it's relationship with other words
#Vectors being multidimentional figures relationships between words
#Tokens are like IDs given to each word
#Embeddings help predict the words around a token. (ONLY MAPS RELATIONS)'''


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

#have to create a client
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60,
)

db = QdrantVectorStore(
    embedding=embeddings,
    client=client,
    collection_name="campus_rag"
)


#db.add_documents(kb_chunks)


#now let's query the DB
""" query="fees for catgeory 1"
result = db.similarity_search(query)
print(result) """


#let's now build an LLM Retreiver

prompt = ChatPromptTemplate.from_template("""
                        The purpose of your creation is for you to clarify all the doubts and questions of students with respect to the context given to you, you are a campus expert about VIT Vellore.
                        People will come asking you doubts regarding anything your job is to be a helpful assisnt and clearly provide them a solid answer which solves their query.
                        For every query you solve perfectly i'll progressively pay you more.

                           -context:
                           {context}

                           -Question:
                           {input}

"""
                           )


brain = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature = 0.7
)


#creating the stuffer chain basically combines the LLM and the Prompt to form the brain
combined_chain=create_stuff_documents_chain(brain,prompt)

#creating the retreiver for LLM
retriever= db.as_retriever(search_kwargs={"k":7})

#creating the RAG chain
rag_chain = create_retrieval_chain(retriever, combined_chain)



#example query
""" response = rag_chain.invoke({"input":"How is life in VIT in 5 sentences?"})
print(response["answer"]) """
