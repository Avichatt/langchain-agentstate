import os
from dotenv import load_dotenv
from typing import TypedDict, List

# LangChain + LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Shared LLM Instance (Groq Llama-3)
llm = ChatGroq(
    model="llama-3.1-8b-instant",  
    api_key=os.getenv("GROQ_API_KEY"),  # Use environment variable for security
    temperature=0
)

# Shared state schema
class AgentState(TypedDict):
    goal: str
    tasks: List[str]
    results: List[str]
    critique: str
    approved: bool
    iteration: int

# Search tool (outside the class)
search = DuckDuckGoSearchRun()

import json
from langchain_core.messages import SystemMessage, HumanMessage

def planner(state: AgentState) -> AgentState:
    system_prompt = (
        "You are a planning agent. You are given a goal and a list of results "
        "from previous iterations. Break the user's goal into at most 5 concepts, actionable tasks."
        "Your task is to generate a list of tasks to "
        "achieve the goal based on the results so far. Respond only with a valid "
        "JSON array of strings. No preamble, no markdown."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Goal: {state['goal']}\nResults: {state['results']}")
    ]

    # Call the LLM
    response = llm.invoke(messages).content.strip()

    try:
        # Clean up any accidental formatting
        clean = (
            response.replace("```json", "")
                    .replace("```", "")
                    .strip()
        )
        tasks = json.loads(clean)
    except json.JSONDecodeError:
        # Fallback: treat raw response as a single task
        tasks = [response]

    print(f"Planner generated tasks: {len(tasks)} tasks:")
    for i, t in enumerate(tasks):
        print(f"{i+1}. {t}")

    # Return updated state
    return {**state, "tasks": tasks}

graph = StateGraph(AgentState)
graph.add_node("planner", planner)
graph.set_entry_point("planner")
graph.add_edge("planner", END)

app = graph.compile()

# run it
if __name__ == "__main__":
    initial_state: AgentState = {
        "goal": "research and summarize the benefits of renewable energy",
        "tasks": [],    
        "results": [],
        "critique": "", 
        "approved": False,
        "iteration": 0  
    }

    final_state = app.invoke(initial_state)
    print("\nFinal State Tasks:")
    for i, task in enumerate(final_state["tasks"]):
        print(f"Task {i+1}: {task}")
    
    print(f"\nExecution completed.")
