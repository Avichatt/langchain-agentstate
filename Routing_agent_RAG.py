import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# ------------------ SETUP ------------------
load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# ------------------ STATE ------------------
class AgentState(TypedDict):
    goal: str
    intent: str
    results: List[str]
    source: str
    final_summary: str

# ------------------ ROUTER ------------------
def router(state: AgentState) -> AgentState:
    query = state["goal"].lower()

    if any(k in query for k in ["2 + 2", "math", "calculate"]):
        intent = "direct_answer"

    elif any(k in query for k in ["week 9", "nlp", "rag", "embeddings"]):
        intent = "retriever_week9"

    elif any(k in query for k in ["week 8", "attention", "query", "key", "value"]):
        intent = "retriever_week8"

    else:
        # fallback to LLM classification
        messages = [
            SystemMessage(content="""
Classify into:
- retriever_week9
- retriever_week8
- direct_answer
Return only label.
"""),
            HumanMessage(content=query)
        ]
        intent = llm.invoke(messages).content.strip()

    print(f"[Router] -> {intent}")

    return {**state, "intent": intent}

def route_decision(state: AgentState) -> str:
    return state["intent"]

# ------------------ RETRIEVERS ------------------
def retriever_week9(state: AgentState) -> AgentState:
    data = """
Week 9:
- Tokenization
- Embeddings
- Transformers
- RAG pipelines
"""
    return {
        **state,
        "results": state["results"] + [data],
        "source": "week9_notes"
    }

def retriever_week8(state: AgentState) -> AgentState:
    data = """
Week 8:
- Attention mechanism
- Query, Key, Value
- Attention weights
"""
    return {
        **state,
        "results": state["results"] + [data],
        "source": "week8_notes"
    }

# ------------------ DIRECT ANSWER ------------------
def direct_answer(state: AgentState) -> AgentState:
    query = state["goal"]

    if "2 + 2" in query:
        answer = "2 + 2 = 4"
    else:
        messages = [
            SystemMessage(content="Answer clearly."),
            HumanMessage(content=query)
        ]
        answer = llm.invoke(messages).content.strip()

    return {
        **state,
        "results": state["results"] + [answer],
        "source": "llm"
    }

# ------------------ SYNTHESIZER ------------------
def synthesizer(state: AgentState) -> AgentState:
    messages = [
        SystemMessage(content="""
Create structured answer:
- Answer
- Key Points
- Source
"""),
        HumanMessage(content=f"""
Query: {state['goal']}
Results: {state['results']}
Source: {state['source']}
""")
    ]

    summary = llm.invoke(messages).content.strip()

    return {**state, "final_summary": summary}

# ------------------ GRAPH ------------------
graph = StateGraph(AgentState)

graph.add_node("router", router)
graph.add_node("retriever_week9", retriever_week9)
graph.add_node("retriever_week8", retriever_week8)
graph.add_node("direct_answer", direct_answer)
graph.add_node("synthesizer", synthesizer)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    route_decision,
    {
        "retriever_week9": "retriever_week9",
        "retriever_week8": "retriever_week8",
        "direct_answer": "direct_answer"
    }
)

graph.add_edge("retriever_week9", "synthesizer")
graph.add_edge("retriever_week8", "synthesizer")
graph.add_edge("direct_answer", "synthesizer")
graph.add_edge("synthesizer", END)

app = graph.compile()

# ------------------ RUN ------------------
if __name__ == "__main__":
    queries = [
        "What did we cover in week 9?",
        "What is 2 + 2?",
        "Explain the attention mechanism"
    ]

    for q in queries:
        print("\n" + "="*50)
        print("Query:", q)

        state = {
            "goal": q,
            "intent": "",
            "results": [],
            "source": "",
            "final_summary": ""
        }

        result = app.invoke(state)

        print("Final Answer:\n", result["final_summary"])