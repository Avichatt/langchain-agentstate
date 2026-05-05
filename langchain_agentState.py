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
    final_summary: str

# Search tool (outside the class)
search = DuckDuckGoSearchRun()

import json
from langchain_core.messages import SystemMessage, HumanMessage

def planner(state: AgentState) -> AgentState:
    print("\n--- [planner] Planning tasks ---")

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

#-------------------------------------------------------------------------------------------------------------------------

# Executor node

def executor(state: AgentState) -> AgentState:
    print(f"\n--- [executor] Executing tasks (Iteration {state['iteration']}) ---")

    results = []
    critique_ctx = ""

    if state["critique"]:
        critique_ctx = f"You previously critiqued the work and asked for the following changes:\n{state['critique']}"

    for task in state["tasks"]:
        # System prompt for execution agent
        system = f"""You are an expert execution agent. Complete the task below thoroughly. Use web search if you nned for current information.{critique_ctx}"""    

        # Try web search for research-based tasks
        search_ctx = ""
        try:
            search_result = search.run(task[:1000])
            search_ctx = f"Found the following search results:\n{search_result[:800]}"
        except Exception as e:
            print(f"[executor] Search failed: {e}")

        # Build messages
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"Task: {task}\n{search_ctx}\n{critique_ctx}")
        ]

        # Call LLM
        result = llm.invoke(messages).content
        results.append(result)

        print(f"\n[executor] Task executed: {task[:60]}...\n  Result: {result[:120]}...")

    # Return updated state
    return {**state, "results": results, "iteration": state["iteration"] + 1}

# --- Synthesizer Node ---
def synthesizer(state: AgentState) -> AgentState:
    print("\n--- Synthesizing Final Summary ---")
    
    system_prompt = (
        "You are a professional technical writer. Your task is to take a goal and a set of "
        "individual research results and synthesize them into a single, cohesive, and "
        "beautifully structured summary. Use Markdown, clear headings, bullet points, "
        "and a concluding section. Ensure the tone is professional and the information is well-organized."
    )
    
    content = f"Goal: {state['goal']}\n\n"
    for i, res in enumerate(state['results']):
        content += f"--- Result Part {i+1} ---\n{res}\n\n"
        
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=content)
    ]
    
    summary = llm.invoke(messages).content
    
    return {**state, "final_summary": summary}

def verifier(state: AgentState) -> AgentState:
    print("\n--- [verifier] Verifying results ---")

    # Safety net: auto-approve after 3 iterations
    if state["iteration"] >= 3:
        print("\n[verifier] Maximum iterations reached. Auto-approving the work.")
        return {**state, "approved": True, "critique": ""}

    # Combine tasks + results for review
    combined_results = "\n\n".join(
        f"Task {i+1}: {t}\nResult: {r}"
        for i, (t, r) in enumerate(zip(state["tasks"], state["results"]))
    )

    system_prompt = (
        "You are a meticulous verifier. Review the tasks and results. "
        "Identify inaccuracies, missing info, or areas for improvement. "
        "Provide a clear critique with specific feedback. If satisfactory, approve. "
        "Use this rubric:\n"
        "completeness: 0-10, accuracy: 0-10, clarity: 0-10.\n"
        "Sum the scores, normalize to 0.0–1.0.\n"
        "Respond ONLY as JSON: {\"score\": 0.85, \"approved\": true, \"critique\": \"...\"}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Goal: {state['goal']}\n\n{combined_results}")
    ]

    raw = llm.invoke(messages).content.strip()

    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        approved = parsed.get("approved", False)
        critique = parsed.get("critique", "")
        score = parsed.get("score", 0.0)
    except Exception:
        approved, critique, score = False, raw, 0.0

    print(f"\n[verifier] Score: {score:.2f}, Approved: {approved}")
    if not approved:
        print(f"[verifier] Critique for improvement:\n{critique}")
    return {**state, "approved": approved, "critique": critique}


#-----------------------------------------------------------------------------------------------------------------------

# --- Build Graph ---
graph = StateGraph(AgentState)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("verifier", verifier)
graph.add_node("synthesizer", synthesizer)

graph.set_entry_point("planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "verifier")

# Conditional edge: If approved, go to synthesizer. Otherwise, loop back to planner for improvement.
def should_continue(state: AgentState):
    if state["approved"]:
        return "synthesizer"
    return "planner"

graph.add_conditional_edges(
    "verifier",
    should_continue,
    {
        "synthesizer": "synthesizer",
        "planner": "planner"
    }
)

graph.add_edge("synthesizer", END)

app = graph.compile()

# --- Run Example ---
if __name__ == "__main__":
    initial_state: AgentState = {
        "goal": "Research and summarize the benefits of renewable energy",
        "tasks": [],
        "results": [],
        "critique": "",
        "approved": False,
        "iteration": 0,
        "final_summary": ""
    }

    final_state = app.invoke(initial_state)
    print("\n" + "="*50)
    print("STRUCTURED SUMMARY")
    print("="*50)
    print(final_state["final_summary"])
