from fastapi import FastAPI
from pydantic import BaseModel


from main import rag_chain

app=FastAPI()

class QuerryRequest(BaseModel):
    query:str



@app.post("/ask")
def ask_question(req: QuerryRequest):
    response = rag_chain.invoke({"input":req.query})
    return {
        "answer": response["answer"]
    }