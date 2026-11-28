import asyncio
import json
import os
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from typing import Literal
from pydantic import BaseModel, Field
from rag import cache, config, history
from rag.chunking import chunk_pdf
from rag.generate import (REFUSAL, citations, clean_citations, should_refuse, standalone_question,
                          stream_answer)
from rag.index import Index

app = FastAPI(title="Document Q&A")
index = Index()


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    source: str | None = None
    history: list[Turn] = Field(default_factory=list)  # earlier messages in this chat


@app.get("/documents")
async def documents():
    return await asyncio.to_thread(index.sources)


@app.post("/documents")
async def upload(file: UploadFile):
    name = os.path.basename(file.filename or "")
    data = await file.read()
    if not name.lower().endswith(".pdf") or not data.startswith(b"%PDF"):
        raise HTTPException(400, "Please upload a PDF file.")

    def ingest():
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(data)
            tmp.flush()
            chunks = chunk_pdf(tmp.name, name, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        index.add(chunks)
        return len(chunks)

    try:
        count = await asyncio.to_thread(ingest)  # CPU work off the event loop
    except ValueError as e:
        raise HTTPException(422, str(e))
    await cache.clear()
    return {"source": name, "chunks": count}


@app.delete("/documents")
async def delete_document(source: str):
    if source not in await asyncio.to_thread(index.sources):
        raise HTTPException(404, "No such document.")
    await asyncio.to_thread(index.delete, source)
    await cache.clear()  # cached answers may quote the deleted document
    return {"deleted": source}


def with_history(action, *args):
    try:
        return action(*args)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/history/{chat_id}")
def get_history(chat_id: str):
    return with_history(history.load, chat_id)


@app.put("/history/{chat_id}")
def put_history(chat_id: str, messages: list[dict]):
    with_history(history.save, chat_id, messages)
    return {"saved": len(messages)}


@app.delete("/history/{chat_id}")
def delete_history(chat_id: str):
    with_history(history.clear, chat_id)
    return {"cleared": True}


def event(kind, data):
    return f"data: {json.dumps({'type': kind, 'data': data})}\n\n"


@app.post("/ask")
async def ask(body: Ask):
    async def events():
        question = await standalone_question(body.question, [t.model_dump() for t in body.history])
        if question != body.question:
            yield event("query", question)
        # keyed on the rewritten question, so "why?" in two different chats cannot share an answer
        k = cache.key(question.strip().lower(), body.source, config.LLM_MODEL,
                      config.LLM_FALLBACK, config.GEMINI_MODEL, config.TOP_K, config.MIN_SCORE)
        cached = await cache.get(k)
        if cached:
            yield event("token", cached["answer"])
            yield event("sources", cached["sources"])
            return
        hits = await asyncio.to_thread(index.search, question, source=body.source)
        if should_refuse(hits):
            yield event("token", REFUSAL)
            yield event("sources", [])
            return
        answer = ""
        try:
            async for token in clean_citations(stream_answer(question, hits)):
                answer += token
                yield event("token", token)
        except Exception as e:  # rate limits, missing key, network
            yield event("error", f"LLM call failed: {type(e).__name__}")
            return
        sources = [] if REFUSAL in answer else citations(hits)
        yield event("sources", sources)
        await cache.put(k, {"answer": answer, "sources": sources})

    return StreamingResponse(events(), media_type="text/event-stream")
