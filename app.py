from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from langchain.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    model = body.get("model", "llama3.1")
    if isinstance(model, dict):  # <-- unwrap dict from frontend
        model = model.get("id", "llama3.1")

    messages = body.get("messages", [])
    prompt = body.get("prompt") or "You are a helpful assistant."
    temperature = body.get("temperature", 0.7)

    # Convert messages into LangChain message objects
    lc_messages = []
    if prompt:
        lc_messages.append(SystemMessage(content=prompt))
    for m in messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    llm = ChatOllama(
        model=model,  # <-- now guaranteed to be a string
        temperature=temperature,
        streaming=True,
    )

    async def generate():
        async for chunk in llm.astream(lc_messages):
            yield chunk.content

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/models")
async def models():
    return JSONResponse([
        {"id": "llama3.1", "name": "LLaMA 3.1"},
        {"id": "mistral", "name": "Mistral 7B"},
    ])
