# Multi-Agent Diabetes Demo – Agent Hierarchies (Mermaid)

This file contains Mermaid diagrams describing the agents and hierarchies used in the Streamlit multi-agent diabetes case conference demo.

You can render these diagrams in any Markdown viewer that supports Mermaid (e.g. GitHub, VS Code with a Mermaid/Markdown extension, or online Mermaid live editors).

---

## 1. Agent hierarchy (role-based)

```mermaid
flowchart TD
    subgraph Case_Conference["Virtual Diabetes Case Conference"]
        Coordinator["Coordinator Agent\n(Orchestrates and summarizes)"]

        Intake["Intake & Triage Agent\n(Collects info, red flags)"]
        Guideline["Guideline Agent\n(Simplified diabetes guideline logic)"]
        Safety["Safety Agent\n(Screens for contraindications & risks)"]
        Education["Patient Education Agent\n(Explains plan in lay terms)"]
        Workflow["Workflow / Care Coordination Agent\n(Suggests concrete workflow steps)"]
        XGBRisk["XGBoost Risk Scoring Agent\n(Specialised ML model on synthetic data)"]
    end

    Coordinator --> Intake
    Coordinator --> Guideline
    Coordinator --> Safety
    Coordinator --> Education
    Coordinator --> Workflow
    Coordinator --> XGBRisk
```

---

## 2. Main data flow for a single case

```mermaid
flowchart LR
    PatientCase["PatientCase input\n(sliders or synthetic row)"]

    Intake["Intake & Triage Agent"]
    Guideline["Guideline Agent"]
    Safety["Safety Agent"]
    Education["Patient Education Agent"]
    Workflow["Workflow / Care Coordination Agent"]
    XGBRisk["XGBoost Risk Scoring Agent"]
    Coordinator["Coordinator Agent"]

    PatientCase --> Intake
    Intake --> Guideline
    Guideline --> Safety
    Guideline --> Education
    Safety --> Education

    Guideline --> Workflow
    Safety --> Workflow

    PatientCase --> XGBRisk

    Intake --> Coordinator
    Guideline --> Coordinator
    Safety --> Coordinator
    Education --> Coordinator
    Workflow --> Coordinator
    XGBRisk --> Coordinator
```

---

## 3. Example conversation & human-in-the-loop focus

This diagram focuses on the *Guideline vs Safety* negotiation plus the clinician.

```mermaid
sequenceDiagram
    participant Intake as Intake Agent
    participant Guideline as Guideline Agent
    participant Safety as Safety Agent
    participant Coord as Coordinator Agent
    participant Clinician as Human Clinician

    Intake->>Guideline: Provide intake summary & red flags
    Guideline->>Safety: Propose plan & initial HbA1c target
    Safety->>Guideline: Raise safety concerns\n(e.g. CKD, age, hypoglycaemia risk)
    Guideline->>Safety: Adjust plan / agree on relaxed target
    Guideline->>Coord: Updated plan and target
    Safety->>Coord: Safety issues & recommendations
    Coord->>Clinician: Summarized handover & options
    Clinician-->>Coord: Final decision & acceptance/rejection of workflow steps
```
