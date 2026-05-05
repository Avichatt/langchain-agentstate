# 🚀 Production-Grade LangChain Agentic Workflow

This repository implements a sophisticated, multi-agent research and synthesis pipeline using **LangGraph**, **LangChain**, and **Groq**. The system is designed for iterative quality improvement through a Planner-Executor-Verifier-Synthesizer architecture.

## 📊 System Architecture

The following diagram illustrates the production-level flow of data and control within the agentic system, highlighting the internal components of each node.

```mermaid
graph TD
    %% Global Styling
    classDef state fill:#f9f,stroke:#333,stroke-width:2px;
    classDef node fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef tool fill:#fff9c4,stroke:#fbc02d,stroke-width:1px;

    subgraph UserInterface["User Entry Point"]
        Goal[User Goal]
    end

    subgraph GlobalState["Shared Agent State (TypedDict)"]
        direction LR
        S1[goal]
        S2[tasks]
        S3[results]
        S4[critique]
        S5[iteration]
    end

    Goal --> PlannerNode

    subgraph PlannerNode["Planner Agent Node"]
        direction TB
        P1[Goal Decomposition]
        P2[JSON Schema Validation]
        P3[Groq Llama 3.1 8B]
        P1 --> P3 --> P2
    end

    subgraph ExecutorNode["Executor Agent Node"]
        direction TB
        E1[DuckDuckGo Search Tool]
        E2[Contextual Augmentation]
        E3[Groq Llama 3.1 8B]
        E1 --> E2 --> E3
    end

    subgraph VerifierNode["Verifier Agent Node"]
        direction TB
        V1[Quality Rubric Scoring]
        V2[Decision Logic]
        V3[Groq Llama 3.1 8B]
        V1 --> V3 --> V2
    end

    subgraph SynthesizerNode["Synthesizer Agent Node"]
        direction TB
        Sy1[Markdown Engine]
        Sy2[Professional Synthesis]
        Sy3[Groq Llama 3.1 8B]
        Sy1 --> Sy3 --> Sy2
    end

    %% Flow Connections
    PlannerNode -->|"Generated Tasks"| ExecutorNode
    ExecutorNode -->|"Research Results"| VerifierNode
    
    VerifierNode -- "Critique & Retry" --> PlannerNode
    VerifierNode -- "Quality Approved" --> SynthesizerNode
    
    SynthesizerNode --> FinalReport[Final Structured Report]

    %% Applying Classes
    class GlobalState state;
    class PlannerNode,ExecutorNode,VerifierNode,SynthesizerNode node;
    class E1,P3,V3,Sy3 tool;
```

---

## 📝 Component Breakdown

### 1. **Global State (`AgentState`)**
The backbone of the system, acting as a "Short-Term Memory" (STM) shared across all nodes. It maintains source traceability, task lists, and iterative critique history.

### 2. **Planner Agent**
- **Function**: Breaks down high-level user goals into a sequence of atomic, researchable tasks.
- **Tech Stack**: Uses **Groq Llama 3.1** with a strict JSON-only output format to ensure downstream compatibility.

### 3. **Executor Agent**
- **Tools**: Integrated with **DuckDuckGo Search** for real-time web research.
- **Function**: Iterates through the plan, fetches external context, and uses the LLM to generate grounded, fact-based answers for each sub-task.

### 4. **Verifier Agent (Quality Control)**
- **Function**: Acts as a "Human-in-the-loop" simulator. It scores the work based on *completeness*, *accuracy*, and *clarity*.
- **Logic**: If the score is below the threshold, it generates a `critique` and loops back to the Planner for a refined approach.

### 5. **Synthesizer Agent**
- **Function**: The final polish. It takes fragmented research results and weaves them into a professional, executive-level Markdown summary.

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Avichatt/langchain-agentstate.git
   ```
2. **Install Dependencies**:
   ```bash
   pip install langgraph langchain-groq python-dotenv duckduckgo-search ddgs
   ```
3. **Environment Configuration**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```
4. **Run the System**:
   ```bash
   python langchain_agentState.py
   ```
