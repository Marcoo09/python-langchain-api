from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import os

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # adjust for frontend
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
    loader = CSVLoader(file_path="student_exam_scores.csv")
    docs = loader.load()

    from langchain_core.documents import Document

    clean_docs = []
    for d in docs:
        if d.page_content is None:
            continue
        content_str = str(d.page_content).strip()
        if not content_str:
            continue
        clean_docs.append(Document(page_content=content_str, metadata=d.metadata))

    vectorstore = FAISS.from_documents(clean_docs, embeddings)
    vectorstore.save_local(faiss_path)

retriever = vectorstore.as_retriever()
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

    # get the *last* user message as the new "question"
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
