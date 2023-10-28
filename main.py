from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect
from pydantic import BaseModel
from typing import List
from qaservice import QAService
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict

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

connected_room = defaultdict(list)

@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    await websocket.accept()
    connected_room[room_id].append(websocket)
    try:
        while True:
            data = await websocket.receive()
            print(data)
            print(type(data))

            # vector store 에 답변 정보 추가
            qa_service.add_admin_answer_info(room_id, "data", "")

            # 메시지를 받았을 때 모든 연결된 클라이언트에게 broadcast 합니다.
            for client in connected_room[room_id]:
                await client.send_text(data)
    except WebSocketDisconnect:
        # 연결이 닫힌 경우 클라이언트를 connected_clients에서 제거합니다.
        # await websocket.close()
        connected_room[room_id].remove(websocket)