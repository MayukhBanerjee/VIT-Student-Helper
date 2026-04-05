from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from main import rag_chain

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuerryRequest(BaseModel):
    query:str



@app.post("/ask")
def ask_question(req: QuerryRequest):
    response = rag_chain.invoke({"input":req.query})
    return {
        "answer": response["answer"]
    }