pytest_plugins = ("pytest_asyncio",)

import pytest
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
import pytest_asyncio

# Import your retriever setup (assuming it's in app.py)
from app import retriever, qa_prompt

@pytest_asyncio.fixture
async def chain():
    """Build a fresh QA chain for each test."""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOllama(model="llama3.1", temperature=0),
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )
    return chain

# def test_prompt_renders_correctly():
#     """Ensure the prompt template inserts context and question correctly."""
#     rendered = qa_prompt.format(
#         context="Student S001 studied 5 hours.",
#         question="How many hours did S001 study?"
#     )
#     assert "Student S001 studied 5 hours." in rendered
#     assert "How many hours did S001 study?" in rendered
#     assert "Answer:" in rendered

# @pytest.mark.asyncio
# async def test_chain_answers_from_context(chain):
#     """Check that the chain produces an answer from given docs."""
#     question = "Who achieved the highest exam score?"
#     result = await chain.ainvoke({"question": question})
#     answer = result["answer"]

#     assert isinstance(answer, str)
#     assert len(answer) > 0


@pytest.mark.asyncio
async def test_chain_idk_from_dataset(chain):
    question = "Who is the teacher?"
    result = await chain.ainvoke({"question": question})
    answer = result["answer"]

    print(f"Answer: {answer}")
    assert "i don’t know" in answer.lower() or "don't know" in answer.lower()