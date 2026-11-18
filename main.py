from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver
from agent import create_graph_builder

app = FastAPI(title="Gemini LangGraph Chatbot")

# MemorySaver keeps chat history per thread_id in RAM
graph = create_graph_builder().compile(checkpointer=MemorySaver())

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "user-1"

@app.get("/")
def health():
    return {"status": "Live!", "docs": "Visit /docs to test the chatbot"}

@app.post("/chat")
def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = graph.invoke({"messages": [("user", req.message)]}, config=config)
    return {
        "thread_id": req.thread_id,
        "reply": result["messages"][-1].content,
    }