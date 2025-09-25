from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import pandas as pd

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Setup embeddings + retriever ---
embeddings = OllamaEmbeddings(model="llama3.1")
faiss_path = "faiss_index"

if os.path.exists(faiss_path):
    vectorstore = FAISS.load_local(
        faiss_path, embeddings, allow_dangerous_deserialization=True
    )
else:
    # ✅ Load CSV row by row
    df = pd.read_csv("student_exam_scores.csv")

    docs = []
    for _, row in df.iterrows():
        if pd.isna(row.get("student_id")):
            continue  # skip empty rows

        content = (
            f"Student {row['student_id']} studied {row['hours_studied']} hours, "
            f"slept {row['sleep_hours']} hours, had {row['attendance_percent']}% attendance, "
            f"previous score {row['previous_scores']}, "
            f"exam score {row['exam_score']}."
        )

        docs.append(Document(page_content=content, metadata=row.to_dict()))

    print(f"✅ Loaded {len(docs)} student rows into FAISS")

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(faiss_path)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOllama(model="llama3.1", temperature=0),
    retriever=retriever,
    memory=memory,
)


@app.post("/chat")
async def chat(req: Request):
    body = await req.json()

    # unwrap model
    model = body.get("model", "llama3.1")
    if isinstance(model, dict):
        model = model.get("id", "llama3.1")

    messages = body.get("messages", [])
    prompt = body.get("prompt") or "You are a helpful assistant."
    temperature = body.get("temperature", 0.7)

    # take the last user message as the "question"
    user_messages = [m["content"] for m in messages if m["role"] == "user"]
    question = user_messages[-1] if user_messages else ""

    async def generate():
        try:
            result = await qa_chain.ainvoke({"question": question})
            yield result["answer"]
        except Exception as e:
            yield f"[ERROR: {repr(e)}]"

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/models")
async def models():
    return JSONResponse(
        [
            {"id": "llama3.1", "name": "LLaMA 3.1"},
            {"id": "mistral", "name": "Mistral 7B"},
        ]
    )
