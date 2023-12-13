import asyncio
from contextlib import asynccontextmanager
from multiprocessing import Process, Manager
import os
import sys
import time
import uuid
from typing import List, Dict

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.responses import Response
from fastapi.security import APIKeyHeader
import uvicorn
from pydantic import BaseModel

import lit.generate as generate
from lit.oai_protocol import ChatCompletionRequest, ChatCompletionResponse, \
    ChatCompletionResponseChoice, ChatMessage, UsageInfo

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
                request = request_buffer.pop(uid)
            except (StopIteration, KeyError):
                time.sleep(0.1)
                continue

            answer, usage = chat.generate_with_context(request.messages, request.temperature)

            response_buffer[uid] = {
                "text": answer,
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": usage["prompt_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "completion_tokens": usage["total_tokens"] - usage["prompt_tokens"]
                }}


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


@app.post('/v1/chat/completions', dependencies=[Depends(api_key_auth)])
async def chat(request: ChatCompletionRequest, background_tasks: BackgroundTasks):
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming not currently supported")

    if request.stop is not None:
        raise HTTPException(status_code=400, detail="Parameter stop not currently supported")

    if request.frequency_penalty:
        raise HTTPException(status_code=400, detail="Parameter frequency_penalty not currently supported")

    if request.presence_penalty:
        raise HTTPException(status_code=400, detail="Parameter presence_penalty not currently supported")

    if request.max_tokens is not None:
        raise HTTPException(status_code=400, detail="Parameter max_tokens not currently supported")

    if request.top_p != 1.0:
        raise HTTPException(status_code=400, detail="Parameter top_p not currently supported")

    uids = [uuid.uuid4() for _ in range(request.n)]

    for uid in uids:
        app.request_buffer[uid] = request
        background_tasks.add_task(cleanup, app.request_buffer, uid)

    waiting_uids = set(uids)
    responses = []

    while waiting_uids:
        await asyncio.sleep(0.1)
        for uid in uids:
            if uid in app.response_buffer:
                response = app.response_buffer.pop(uid)
                responses.append(response)
                waiting_uids.remove(uid)

    choices = []
    chat_completions = []

    usage = UsageInfo()
    for i, response in enumerate(responses):
        choices.append(
            ChatCompletionResponseChoice(
                index=i,
                message=ChatMessage(role="assistant", content=response["text"]),
                finish_reason=response.get("finish_reason", "stop"),
            )
        )
        task_usage = UsageInfo.parse_obj(response["usage"])
        for usage_key, usage_value in task_usage.dict().items():
            setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    model = request.model or "lit-gpt"
    return ChatCompletionResponse(model=model, choices=choices, usage=usage)


def serve(llm, temperature, device_ids, port=8000, timeout_keep_alive=30, blocking=True):
    if not blocking:
        raise NotImplementedError

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    app.llm = llm
    app.temperature = temperature
    app.device_ids = device_ids

    uvicorn.run(host="0.0.0.0", port=port, app=app, timeout_keep_alive=timeout_keep_alive)


if __name__ == "__main__":
    app.model_name = "microsoft/phi-1_5"
    app.device_ids = [0]

    if len(sys.argv) > 1:
        app.device_ids = [int(el) for el in sys.argv[2:]]

    uvicorn.run(host="0.0.0.0", port=8000, app=app, timeout_keep_alive=30)
