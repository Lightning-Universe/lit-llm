import asyncio
from contextlib import asynccontextmanager
from multiprocessing import Process, Manager
import os
import sys
import time
import uuid

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.responses import Response
from fastapi.security import APIKeyHeader
import uvicorn
from pydantic import BaseModel

import lit.generate as generate


X_API_KEY = APIKeyHeader(name='X-API-Key')

LIT_SERVER_API_KEY = os.environ.get("LIT_SERVER_API_KEY", "1234567890")


class ChatRequest(BaseModel):
    prompt: str
    temperature: float


def inference_worker(llm, temperature, device_id, worker_id, request_buffer, response_buffer):
    with llm.chat(temperature=temperature) as chat:
        print(f"Done loading, server ready", file=sys.stderr)

        while True:
            try:
                uid = next(iter(request_buffer.keys()))
                chat_req = request_buffer.pop(uid)
            except (StopIteration, KeyError):
                time.sleep(0.1)
                continue

            response_buffer[uid] = "".join(list(chat.stream(chat_req.prompt)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager = Manager()
    app.request_buffer = manager.dict()
    app.response_buffer = manager.dict()

    for worker_id, device_id in enumerate(app.device_ids):
        process = Process(
            target=inference_worker,
            args=(app.llm, app.temperature, device_id, worker_id, app.request_buffer, app.response_buffer),
            daemon=True)
        process.start()

    yield


app = FastAPI(lifespan=lifespan)


def api_key_auth(x_api_key: str = Depends(X_API_KEY)):
    if x_api_key != LIT_SERVER_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key. Check that you are passing a correct 'X-API-Key' in your header."
        )


def cleanup(request_buffer, uid):
    try:
        request_buffer.pop(uid)
    except KeyError:
        pass


@app.post('/chat', dependencies=[Depends(api_key_auth)])
async def chat(chat_req: ChatRequest, background_tasks: BackgroundTasks):
    uid = uuid.uuid4()
    app.request_buffer[uid] = chat_req

    background_tasks.add_task(cleanup, app.request_buffer, uid)

    while True:
        await asyncio.sleep(0.1)
        if uid in app.response_buffer:
            response = app.response_buffer.pop(uid)
            break

    return Response(response, media_type='text/plain')


def serve(llm, temperature, device_ids, port=8000, timeout_keep_alive=20, blocking=True):
    if not blocking:
        raise NotImplementedError

    app.llm = llm
    app.temperature = temperature
    app.device_ids = device_ids

    uvicorn.run(host="0.0.0.0", port=port, app=app, timeout_keep_alive=timeout_keep_alive)


if __name__ == "__main__":
    app.model_name = "microsoft/phi-1_5"
    app.device_ids = [0]

    if len(sys.argv) > 1:
        app.device_ids = [int(el) for el in sys.argv[2:]]

    uvicorn.run(host="0.0.0.0", port=8000, app=app, timeout_keep_alive=20)
