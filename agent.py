from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_retries=3,
)

def create_graph_builder():
    builder = StateGraph(MessagesState)
    builder.add_node("chat", lambda state: {"messages": [llm.invoke(state["messages"])]})
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder