from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect
from pydantic import BaseModel
from typing import List
from qaservice import QAService
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
qa_service = QAService()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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


class Chat(BaseModel):
    commentType: str
    idNo: str
    commentNo: int
    nickname: str
    message: str

class Item(BaseModel):
    list: List[Chat]
    next: int

idx = 0
@app.post("/classifier_api/classify")
def classify(item : Item):
    global idx
    chat_list = item.list

    result = dict()
    result["chat_data"] = list()

    for chat_idx in range(len(chat_list)):
        idx += 1
        chat = chat_list[chat_idx]
        classifier_result = dict()
        classifier_result["commentNo"] = chat.commentNo
        classifier_result["nickname"] = chat.nickname
        classifier_result["message"] = chat.message

        if idx % 3 == 0:
            classifier_result["result"] = "일반"
        elif idx % 3 == 1:
            classifier_result["result"] = "질문"
        else:
            classifier_result["result"] = "요청"

        result["chat_data"].append(classifier_result)

    return result

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        await websocket.close()