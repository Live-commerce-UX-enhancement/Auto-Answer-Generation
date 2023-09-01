from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from qaservice import QAService
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
qa_service = QAService()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BroadcastInformation(BaseModel):
    type: str
    texts: List[str]


class ProductInformation(BaseModel):
    id: str
    name: str
    texts: List[str]


class Information(BaseModel):
    broadcast: List[BroadcastInformation]
    product: List[ProductInformation]


@app.post("/{broadcast_id}/detail")
def add_info(broadcast_id, information: Information):
    qa_service.add_info(broadcast_id, information)


@app.get("/{broadcast_id}/query")
def get_answer(broadcast_id, q: str):
    return qa_service.get_answer(broadcast_id, q)


@app.get("/ping")
def pingPong():
    return "pong"
