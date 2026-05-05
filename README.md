# 🤖 AI Multi-Agent Research Assistant

**Goal**: Deliver accurate, well-researched, and well-structured answers through iterative planning, execution, verification, and synthesis.

---

## 🏛️ High-Level Architecture

This system leverages a state-of-the-art multi-agent design, utilizing **LangGraph** for orchestration, **Groq Llama 3.1** for reasoning, and **DuckDuckGo** for real-time information retrieval.

```mermaid
graph TD
    %% Theme: Dark / Professional
    accTitle: AI Multi-Agent Research Assistant Architecture
    accDescr: A detailed flowchart showing the interaction between Global State, Planner, Executor, Verifier, and Synthesizer agents.

    %% Global Styling
    classDef state fill:#1a1a1a,stroke:#4a90e2,stroke-width:2px,color:#fff;
    classDef agent fill:#2c2c2c,stroke:#50e3c2,stroke-width:2px,color:#fff;
    classDef decision fill:#333,stroke:#f5a623,stroke-width:2px,color:#fff;
    classDef tool fill:#1a1a1a,stroke:#d0021b,stroke-width:1px,color:#ccc,stroke-dasharray: 5 5;
    classDef output fill:#1a1a1a,stroke:#7ed321,stroke-width:2px,color:#fff;

    %% Global State (STM)
    subgraph GlobalState["1. GLOBAL STATE (AgentState) - Short-Term Memory (STM)"]
        direction LR
        StateInfo["• goal: str<br/>• plan: List[Task]<br/>• iteration: int<br/>• results: List[str]<br/>• critique: str<br/>• approved: bool"]
    end

    %% User Input
    UserInput([User Input: High-level Goal]) --> PlannerAgent

    %% Planner Agent
    subgraph PlannerNode["2. PLANNER AGENT"]
        direction TB
        P_Func["Function: Breaks down goals into<br/>atomic, researchable tasks."]
        P_Stack["Tech Stack: Groq Llama 3.1"]
        P_Out["Output: Strict JSON (Plan)"]
    end

    %% Executor Agent
    subgraph ExecutorNode["3. EXECUTOR AGENT"]
        direction TB
        E_Tools["Tools: DuckDuckGo Search"]
        E_Func["Function: Fetches real-time web<br/>context and generates answers."]
        E_Out["Output: Research result per task"]
    end

    %% Verifier Agent
    subgraph VerifierNode["4. VERIFIER AGENT (Quality Control)"]
        direction TB
        V_Func["Function: Scores work on<br/>completeness, accuracy, and clarity."]
        V_Logic["Logic: Generate critique if<br/>score < threshold."]
        V_Out["Output: Score + Critique"]
    end

    %% Synthesizer Agent
    subgraph SynthesizerNode["5. SYNTHESIZER AGENT"]
        direction TB
        S_Func["Function: Weaves verified results<br/>into a professional report."]
        S_Out["Output: Polished Markdown Summary"]
    end

    %% Decision Logic
    Decision{Score ≥ Threshold?}

    %% Connections
    GlobalState -.->|Read/Write| PlannerNode
    GlobalState -.->|Read/Write| ExecutorNode
    GlobalState -.->|Read/Write| VerifierNode
    GlobalState -.->|Read/Write| SynthesizerNode

    PlannerNode --> ExecutorNode
    ExecutorNode --> VerifierNode
    VerifierNode --> Decision

    Decision -- "NO (Needs Improvement)" --> PlannerNode
    Decision -- "YES (Approved)" --> SynthesizerNode

    SynthesizerNode --> FinalOutput[[FINAL OUTPUT: Executive Summary]]

    %% External Tools
    DDG{{DuckDuckGo Search Tool}} -.->|External Context| ExecutorNode

    %% Applying Classes
    class GlobalState state;
    class PlannerNode,ExecutorNode,VerifierNode,SynthesizerNode agent;
    class Decision decision;
    class DDG tool;
    class FinalOutput output;
```

---

## 🛠️ Agent Responsibilities

### 1. **Global State (STM)**
Acts as the "Source of Truth" shared across all agents. It stores the user's objective, the generated research plan, raw results, and the iterative feedback loop history.

### 2. **Planner Agent**
- **Objective**: Strategic decomposition.
- **Output**: A structured list of tasks designed to eliminate ambiguity and maximize research depth.

### 3. **Executor Agent**
- **Objective**: Grounded execution.
- **Capability**: Utilizes the **DuckDuckGo Search** tool to pull live data, ensuring the assistant isn't limited by its training data cutoff.

### 4. **Verifier Agent (The Evaluator)**
- **Objective**: Quality assurance.
- **Logic**: It performs a critical audit of the research. If the content is hallucinated or incomplete, it forces a "Critique Loop" to re-plan and re-execute.

### 5. **Synthesizer Agent**
- **Objective**: Executive presentation.
- **Output**: Transforms fragmented data points into a cohesive, professional Markdown report suitable for high-stakes decision-making.

---

## 🚀 Getting Started

1. **Setup Environment**:
   ```bash
   pip install langgraph langchain-groq python-dotenv duckduckgo-search ddgs
   ```
2. **API Configuration**:
   Add your `GROQ_API_KEY` to a `.env` file.
3. **Execution**:
   ```bash
   python langchain_agentState.py
   ```
