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
from langchain.prompts import PromptTemplate

from langchain.agents import initialize_agent, Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = OllamaEmbeddings(model="llama3.1")
faiss_path = "faiss_index"

if os.path.exists(faiss_path):
    vectorstore = FAISS.load_local(
        faiss_path, embeddings, allow_dangerous_deserialization=True
    )
else:
    df = pd.read_csv("student_exam_scores.csv")

    docs = []
    for _, row in df.iterrows():
        if pd.isna(row.get("student_id")):
            print("Skipping empty row")
            continue  # skip empty rows

        content = (
            f"Student {row['student_id']} studied {row['hours_studied']} hours, "
            f"slept {row['sleep_hours']} hours, had {row['attendance_percent']}% attendance, "
            f"previous score {row['previous_scores']}, "
            f"exam score {row['exam_score']}."
        )

        docs.append(Document(page_content=content, metadata=row.to_dict()))

    avg_score = df["exam_score"].mean()
    max_score = df["exam_score"].max()
    min_score = df["exam_score"].min()
    top_student = df.loc[df["exam_score"].idxmax(), "student_id"]
    worst_student = df.loc[df["exam_score"].idxmin(), "student_id"]
    
    summary_docs = [
        Document(page_content=f"The average exam score across all students is {avg_score:.2f}."),
        Document(page_content=f"The highest exam score is {max_score}, achieved by student {top_student}."),
        Document(page_content=f"The lowest exam score is {min_score}, achieved by student {worst_student}."),
        Document(page_content=f"There are {len(df)} students in the dataset."),
    ]
    docs.extend(summary_docs)

    print(f"✅ Loaded {len(docs)} student rows into FAISS")

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(faiss_path)

retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 30, "fetch_k": 100})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. 
Use the ONLY following context to answer the user's question. 
If the answer is not contained in the context, say "I don’t know".
You should answer in English.
Context:
{context}

Question:
{question}

Answer:"""
)
# Option 1 -> simple chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOllama(model="llama3.1", temperature=0),
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
)
# Option 2 -> agent with tools
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [
    Tool(
        name="Wikipedia",
        func=lambda q: str(wiki.run(q)),
        description="Search Wikipedia for general knowledge like capitals, people, history, etc."
    ),
]

agent = initialize_agent(
    tools,
    ChatOllama(model="llama3.1", temperature=0),
    agent_type="openai-functions",
    verbose=False,
    handle_parsing_errors=True
)

@app.post("/chat")
async def chat(req: Request):
    body = await req.json()

    model = body.get("model", "llama3.1")
    if isinstance(model, dict):
        model = model.get("id", "llama3.1")

    messages = body.get("messages", [])

    user_messages = [m["content"] for m in messages if m["role"] == "user"]
    question = user_messages[-1] if user_messages else ""

    mode = "rag"
    async def generate():
        try:
            if mode == "agent":
                result = await agent.ainvoke({"input": question})
                yield result["output"]
                return  
            else:
                result = await qa_chain.ainvoke({"question": question})
                yield result["answer"]
                return  
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

@app.post("/clear_memory")
async def clear_memory():
    memory.clear()
    return {"status": "ok", "message": "Chat history cleared"}    