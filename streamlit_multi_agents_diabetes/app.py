import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Dict, Any, List


# ---------------------------
# Domain model
# ---------------------------

@dataclass
class PatientCase:
    age: int
    sex: str
    bmi: float
    duration_dm_years: float
    hba1c: float
    eGFR: float
    on_metformin: bool
    on_sulfonylurea: bool
    on_sglt2: bool
    has_cvd: bool
    has_ckd: bool
    hypoglycemia_history: bool
    main_concern: str


@dataclass
class AgentOutput:
    name: str
    role: str
    reasoning: str
    message: str
    structured: Dict[str, Any]


# ---------------------------
# Agent implementations
# ---------------------------

class IntakeTriageAgent:
    name = "Intake & Triage Agent"
    role = "Collects key information, identifies red flags, and frames the main clinical questions."

    def act(self, case: PatientCase) -> AgentOutput:
        red_flags = []

        if case.hba1c >= 10:
            red_flags.append("Very poor glycaemic control (HbA1c ≥ 10%)")
        if case.eGFR < 30:
            red_flags.append("Severely reduced renal function (eGFR < 30 mL/min)")
        if case.hypoglycemia_history:
            red_flags.append("History of significant hypoglycaemia")
        if case.has_cvd:
            red_flags.append("Established cardiovascular disease")
        if case.has_ckd:
            red_flags.append("Established chronic kidney disease")

        reasoning = (
            "I review basic demographics, glycaemic control, renal function, "
            "comorbidities, and the patient's main concern to identify red flags "
            "and narrow down key clinical questions."
        )

        main_questions = [
            "Is current glycaemic control adequate for this patient?",
            "Is there any urgent safety issue (e.g., severe CKD, hypoglycaemia risk)?",
            "Should we intensify or de-intensify diabetes therapy?",
            "Which class of drugs might be preferred (e.g., SGLT2i, GLP-1 RA, SU)?"
        ]

        message_parts = [
            f"Patient is a {case.age}-year-old {case.sex.lower()} with type 2 diabetes for "
            f"{case.duration_dm_years:.1f} years, BMI {case.bmi:.1f} kg/m².",
            f"Current HbA1c is {case.hba1c:.1f}%. eGFR is {case.eGFR:.0f} mL/min.",
            f"Main concern: {case.main_concern or 'not specified by patient.'}",
        ]

        if red_flags:
            message_parts.append("Identified red flags:")
            for rf in red_flags:
                message_parts.append(f"- {rf}")
        else:
            message_parts.append("No major red flags identified based on the provided data.")

        message_parts.append("Key clinical questions:")
        for q in main_questions:
            message_parts.append(f"- {q}")

        return AgentOutput(
            name=self.name,
            role=self.role,
            reasoning=reasoning,
            message="\n".join(message_parts),
            structured={
                "red_flags": red_flags,
                "main_questions": main_questions,
            },
        )


class GuidelineAgent:
    name = "Guideline Agent"
    role = "Applies simplified diabetes management principles to suggest therapy options."

    def act(self, case: PatientCase, intake_info: AgentOutput) -> AgentOutput:
        reasoning_lines: List[str] = []
        plan: List[str] = []
        targets: Dict[str, Any] = {}

        # Glycaemic target (very simplified, for teaching only)
        if case.age < 65 and not (case.has_cvd or case.has_ckd or case.hypoglycemia_history):
            target = "< 7.0%"
            reasoning_lines.append(
                "Younger patient without major comorbidities → aim for tighter HbA1c (<7%)."
            )
        else:
            target = "< 7.5–8.0%"
            reasoning_lines.append(
                "Older and/or with comorbidities or hypoglycaemia risk → slightly more relaxed target."
            )

        targets["hba1c_target"] = target

        # Therapy intensification logic (simplified)
        if case.hba1c <= 7.0:
            plan.append("Glycaemic control is near or at target. Focus on lifestyle optimization and monitoring.")
        elif 7.0 < case.hba1c <= 9.0:
            reasoning_lines.append("HbA1c moderately above target → consider intensifying therapy.")
            if not case.on_metformin and case.eGFR >= 30:
                plan.append("Start or optimize metformin (if not contraindicated).")
            else:
                plan.append("Optimize metformin dose, if tolerated.")
            if case.has_cvd or case.has_ckd:
                plan.append("Add an SGLT2 inhibitor or GLP-1 RA with proven CV/renal benefit.")
            else:
                plan.append("Consider adding SGLT2 inhibitor, GLP-1 RA, or DPP-4i based on patient factors.")
        else:
            reasoning_lines.append("HbA1c markedly above target → intensify therapy more aggressively.")
            plan.append("Ensure metformin at maximally tolerated dose (if eGFR allows).")
            if case.has_cvd or case.has_ckd:
                plan.append("Prioritize SGLT2 inhibitor and/or GLP-1 RA with CV/renal benefit.")
            plan.append("Consider triple therapy and evaluate need for basal insulin in shared decision-making.")

        # Comorbidity-specific recommendations
        if case.has_cvd:
            plan.append("Because of established CVD, emphasize agents with CV benefit (SGLT2i, GLP-1 RA).")
        if case.has_ckd and case.eGFR < 60:
            plan.append("Because of CKD, prefer SGLT2i with renal benefit if eGFR is adequate, and avoid nephrotoxic drugs.")

        reasoning = "I apply a simplified, didactic version of type 2 diabetes guidelines:\n- " + "\n- ".join(reasoning_lines)

        message_parts = [
            f"Target HbA1c for this patient: {target} (for teaching only, not a strict rule).",
            "",
            "Suggested management plan (simplified):",
        ]
        for step in plan:
            message_parts.append(f"- {step}")

        return AgentOutput(
            name=self.name,
            role=self.role,
            reasoning=reasoning,
            message="\n".join(message_parts),
            structured={
                "hba1c_target": target,
                "plan_steps": plan,
            },
        )


class SafetyAgent:
    name = "Safety Agent"
    role = "Screens for basic safety issues and potential contraindications (simplified)."

    def act(self, case: PatientCase, guideline_info: AgentOutput) -> AgentOutput:
        issues: List[str] = []
        recommendations: List[str] = []

        if case.eGFR < 30:
            issues.append("Severely reduced eGFR (<30 mL/min); metformin and some SGLT2i may be contraindicated.")
            recommendations.append("Review metformin dose; consider stopping if eGFR <30 mL/min.")

        if case.hypoglycemia_history and case.on_sulfonylurea:
            issues.append("History of hypoglycaemia while on sulfonylurea.")
            recommendations.append("Consider reducing or discontinuing sulfonylurea; prefer agents with lower hypoglycaemia risk.")

        if case.age > 75 and case.hba1c < 7.0:
            issues.append("Tight control in an older adult (>75 years) could carry hypoglycaemia risk.")
            recommendations.append("Consider relaxing HbA1c target and possibly de-intensifying therapy.")

        if not issues:
            issues.append("No major safety issues identified with the limited information provided.")
            recommendations.append("Continue to monitor renal function, hypoglycaemia risk, and polypharmacy.")

        reasoning = (
            "I scan for simple but high-yield safety concerns: renal function, hypoglycaemia risk, "
            "and overtreatment in older adults. This is intentionally simplified for teaching."
        )

        message_parts = ["Safety considerations:"]
        for i in issues:
            message_parts.append(f"- {i}")
        message_parts.append("")
        message_parts.append("Safety recommendations:")
        for r in recommendations:
            message_parts.append(f"- {r}")

        return AgentOutput(
            name=self.name,
            role=self.role,
            reasoning=reasoning,
            message="\n".join(message_parts),
            structured={
                "safety_issues": issues,
                "safety_recommendations": recommendations,
            },
        )


class PatientEducationAgent:
    name = "Patient Education Agent"
    role = "Translates the plan into patient-friendly language and key counseling points."

    def act(
        self,
        case: PatientCase,
        guideline_info: AgentOutput,
        safety_info: AgentOutput,
    ) -> AgentOutput:
        target = guideline_info.structured.get("hba1c_target", "individualised target")
        plan_steps = guideline_info.structured.get("plan_steps", [])
        safety_rec = safety_info.structured.get("safety_recommendations", [])

        reasoning = (
            "I convert the technical plan into accessible language, focusing on what the patient "
            "needs to understand about their goals, medications, and safety precautions."
        )

        message_parts = [
            "How I would explain this to the patient:",
            "",
            f"- Your current long-term sugar marker (HbA1c) is {case.hba1c:.1f}%. For you, "
            f"we are aiming for about {target}. This is a flexible goal that we adapt together.",
            "",
            "Treatment focus:",
        ]

        if plan_steps:
            for step in plan_steps:
                message_parts.append(f"- {step}")
        else:
            message_parts.append("- Continue current therapy and healthy lifestyle.")

        if safety_rec:
            message_parts.append("")
            message_parts.append("Safety and monitoring (in plain language):")
            for r in safety_rec:
                message_parts.append(f"- {r}")

        message_parts.append("")
        message_parts.append(
            "Please remember: these are teaching examples and in real life, we decide together "
            "what fits your situation best."
        )

        return AgentOutput(
            name=self.name,
            role=self.role,
            reasoning=reasoning,
            message="\n".join(message_parts),
            structured={},
        )


class CoordinatorAgent:
    name = "Coordinator Agent"
    role = "Orchestrates the other agents and provides a structured summary."

    def act(
        self,
        case: PatientCase,
        intake: AgentOutput,
        guideline: AgentOutput,
        safety: AgentOutput,
        education: AgentOutput,
    ) -> AgentOutput:
        reasoning = (
            "I integrate the outputs from all agents into a brief, structured handover summary. "
            "This shows how multi-agent collaboration can support clinical reasoning, while the clinician remains in charge."
        )

        message_parts = [
            "Coordinator summary (for clinician):",
            "",
            "1. Intake & Triage highlights:",
            *[f"   {line}" for line in intake.message.splitlines()],
            "",
            "2. Guideline-based suggestions:",
            *[f"   {line}" for line in guideline.message.splitlines()],
            "",
            "3. Safety considerations:",
            *[f"   {line}" for line in safety.message.splitlines()],
            "",
            "4. Patient-friendly explanation (to adapt for your own consultation):",
            *[f"   {line}" for line in education.message.splitlines()],
        ]

        return AgentOutput(
            name=self.name,
            role=self.role,
            reasoning=reasoning,
            message="\n".join(message_parts),
            structured={
                "intake": asdict(intake),
                "guideline": asdict(guideline),
                "safety": asdict(safety),
                "education": asdict(education),
            },
        )


class WorkflowAgent:
    name = "Workflow / Care Coordination Agent"
    role = "Suggests concrete workflow steps (orders, referrals, follow-up) based on the plan."

    def act(self, case: PatientCase, guideline_info: AgentOutput, safety_info: AgentOutput) -> AgentOutput:
        tasks = []

        hba1c_target = guideline_info.structured.get("hba1c_target", "individualised target")
        has_ckd = case.has_ckd or case.eGFR < 60

        tasks.append("Schedule follow-up visit in 3–6 months to reassess HbA1c and symptoms.")
        tasks.append("Arrange lifestyle counselling (nutrition / physical activity) if available.")

        if has_ckd:
            tasks.append("Ensure regular monitoring of renal function (eGFR, albuminuria).")

        if case.has_cvd:
            tasks.append("Review cardiovascular risk management (BP, lipids, antiplatelet therapy as appropriate).")

        tasks.append(f"Document agreed illustrative HbA1c target (~{hba1c_target}) in the record.")
        tasks.append("Generate a draft physician letter / Entlassbrief summarising the case, plan, and key safety considerations.")

        reasoning = (
            "I translate the abstract plan into simple workflow steps such as scheduling follow-up, "
            "ordering monitoring tests, and documenting the agreed (illustrative) target."
        )

        message_parts = ["Proposed workflow steps (for teaching only):"]
        for t in tasks:
            message_parts.append(f"- {t}")

        return AgentOutput(
            name=self.name,
            role=self.role,
            reasoning=reasoning,
            message="\n".join(message_parts),
            structured={"tasks": tasks},
        )


class XGBoostRiskAgent:
    name = "XGBoost Risk Scoring Agent"
    role = "Uses a small XGBoost model to estimate an illustrative complication risk (teaching only)."

    def __init__(self, booster: xgb.Booster) -> None:
        self.booster = booster

    def act(self, case: PatientCase) -> AgentOutput:
        features = np.array([[case.age, case.bmi, case.hba1c, case.eGFR]], dtype=float)
        dmatrix = xgb.DMatrix(features)
        proba = float(self.booster.predict(dmatrix)[0])

        reasoning = (
            "I use a tiny XGBoost decision tree model trained on synthetic data with features "
            "(age, BMI, HbA1c, eGFR) to estimate an *illustrative* probability that this patient "
            "belongs to a 'higher complication risk' group. This is purely for demonstration and "
            "must not be used for real risk prediction."
        )

        message_parts = [
            "Illustrative XGBoost-based risk score (synthetic model):",
            f"- Estimated probability of being in the 'higher risk' group: **{proba:.2f}**",
            "",
            "This model was trained on synthetic data with a handcrafted rule (higher HbA1c, lower eGFR, "
            "and older age → higher label). The purpose is to show how a specialised ML model can be wrapped "
            "inside an agent, not to provide clinically valid prediction.",
        ]

        return AgentOutput(
            name=self.name,
            role=self.role,
            reasoning=reasoning,
            message="\n".join(message_parts),
            structured={"risk_probability": proba},
        )


# ---------------------------
# Streamlit UI
# ---------------------------

@st.cache_resource
def load_xgb_risk_model() -> xgb.Booster:
    """Train a tiny synthetic XGBoost model for didactic risk scoring.

    Features: age, BMI, HbA1c, eGFR.
    Label: synthetic 'higher risk' flag based on simple rules with noise.
    """
    rng = np.random.default_rng(42)
    n = 1000
    age = rng.integers(30, 90, size=n)
    bmi = rng.uniform(20.0, 40.0, size=n)
    hba1c = rng.uniform(6.0, 11.0, size=n)
    egfr = rng.uniform(20.0, 100.0, size=n)

    # Handcrafted synthetic label: higher risk if clearly poor control or low eGFR
    base_risk = (
        (hba1c >= 8.5).astype(float)
        + (egfr < 45).astype(float)
        + (age > 75).astype(float)
    )
    noise = rng.normal(0, 0.3, size=n)
    logits = base_risk + noise
    prob = 1 / (1 + np.exp(-logits))
    y = (prob > 0.5).astype(int)

    X = np.column_stack([age, bmi, hba1c, egfr])
    dtrain = xgb.DMatrix(X, label=y)

    params = {
        "max_depth": 3,
        "eta": 0.2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    }

    booster = xgb.train(params=params, dtrain=dtrain, num_boost_round=10)
    return booster

@st.cache_data
def load_synthetic_diabetes(path: str = "../data/synthetic_diabetes.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.reset_index(drop=True)
    return df


def patient_case_from_row(row: pd.Series) -> PatientCase:
    age = int(row.get("PatientAge", 58))
    bmi_val = row.get("BodyMassIndex")
    bmi = float(bmi_val) if pd.notna(bmi_val) else 28.0
    bg = row.get("BloodGlucose", 140)
    bg_val = float(bg) if pd.notna(bg) else 140.0
    hba1c = max(5.5, min(14.0, round(bg_val / 15.0, 1)))
    egfr_base = 90.0 - max(0, age - 40) * 0.8
    eGFR = max(20.0, min(100.0, egfr_base))
    has_ckd = eGFR < 60
    diabetes_status = int(row.get("DiabetesStatus", 1))
    has_cvd = bool(diabetes_status and age >= 60)
    on_metformin = bool(diabetes_status)
    on_sulfonylurea = False
    on_sglt2 = bool(diabetes_status and eGFR >= 45)
    hypoglycemia_history = False
    concern = "Synthetic dataset patient: worried about long-term complications."

    return PatientCase(
        age=age,
        sex="Male" if age % 2 == 0 else "Female",
        bmi=bmi,
        duration_dm_years=max(0.0, round((age - 30) / 5.0, 1)),
        hba1c=hba1c,
        eGFR=eGFR,
        on_metformin=on_metformin,
        on_sulfonylurea=on_sulfonylurea,
        on_sglt2=on_sglt2,
        has_cvd=has_cvd,
        has_ckd=has_ckd,
        hypoglycemia_history=hypoglycemia_history,
        main_concern=concern,
    )


def build_patient_case() -> PatientCase:
    st.sidebar.header("Patient characteristics")

    age = st.sidebar.slider("Age (years)", 30, 90, 58)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female", "Other"])
    bmi = st.sidebar.slider("BMI (kg/m²)", 18.0, 45.0, 31.5, 0.1)
    duration = st.sidebar.slider("Duration of type 2 diabetes (years)", 0.0, 30.0, 8.0, 0.5)
    hba1c = st.sidebar.slider("HbA1c (%)", 5.5, 14.0, 8.9, 0.1)
    egfr = st.sidebar.slider("eGFR (mL/min)", 10, 120, 65)

    st.sidebar.subheader("Current therapy")
    on_metformin = st.sidebar.checkbox("On metformin", True)
    on_su = st.sidebar.checkbox("On sulfonylurea", False)
    on_sglt2 = st.sidebar.checkbox("On SGLT2 inhibitor", False)

    st.sidebar.subheader("Comorbidities / risks")
    has_cvd = st.sidebar.checkbox("Established cardiovascular disease", False)
    has_ckd = st.sidebar.checkbox("Chronic kidney disease", egfr < 60)
    hypo_hist = st.sidebar.checkbox("History of significant hypoglycaemia", False)

    concern = st.sidebar.text_area(
        "Patient's main concern (free text, teaching only)",
        "Worried about long-term complications and tiredness.",
    )

    return PatientCase(
        age=age,
        sex=sex,
        bmi=bmi,
        duration_dm_years=duration,
        hba1c=hba1c,
        eGFR=egfr,
        on_metformin=on_metformin,
        on_sulfonylurea=on_su,
        on_sglt2=on_sglt2,
        has_cvd=has_cvd,
        has_ckd=has_ckd,
        hypoglycemia_history=hypo_hist,
        main_concern=concern,
    )


def run_multi_agent_pipeline(case: PatientCase) -> List[AgentOutput]:
    intake_agent = IntakeTriageAgent()
    guideline_agent = GuidelineAgent()
    safety_agent = SafetyAgent()
    education_agent = PatientEducationAgent()
    coordinator_agent = CoordinatorAgent()
    workflow_agent = WorkflowAgent()
    xgb_booster = load_xgb_risk_model()
    xgb_agent = XGBoostRiskAgent(xgb_booster)

    intake = intake_agent.act(case)
    guideline = guideline_agent.act(case, intake)
    safety = safety_agent.act(case, guideline)
    education = education_agent.act(case, guideline, safety)
    coordinator = coordinator_agent.act(case, intake, guideline, safety, education)
    workflow = workflow_agent.act(case, guideline, safety)
    xgb_risk = xgb_agent.act(case)

    # Return agents in an order suitable for display, with Coordinator last
    return [
        intake,
        guideline,
        safety,
        education,
        workflow,
        xgb_risk,
        coordinator,
    ]


def build_guideline_safety_conversation(case: PatientCase) -> (List[tuple], str, str):
    """Construct a didactic conversation between Guideline and Safety agents.

    Returns a list of (speaker, message) turns, the final illustrative HbA1c target,
    and the initial guideline target.
    """
    intake_agent = IntakeTriageAgent()
    guideline_agent = GuidelineAgent()
    safety_agent = SafetyAgent()

    intake = intake_agent.act(case)
    guideline = guideline_agent.act(case, intake)
    safety = safety_agent.act(case, guideline)

    target_initial = guideline.structured.get("hba1c_target", "individualised target")
    safety_issues = safety.structured.get("safety_issues", [])

    turns: List[tuple] = []

    turns.append((
        "Coordinator",
        "Guideline Agent, please propose a plan based on the intake summary (including HbA1c and comorbidities).",
    ))
    turns.append((
        "Guideline Agent",
        f"Based on this patient's profile I suggest a target HbA1c of {target_initial} and intensification according to simplified guidelines.",
    ))

    # Safety responds
    if safety_issues and not (
        len(safety_issues) == 1 and "No major safety issues" in safety_issues[0]
    ):
        turns.append((
            "Safety Agent",
            "From a safety perspective I see the following concerns: " + "; ".join(safety_issues) + ".",
        ))

        relaxed_target = target_initial
        if case.age > 75 or any("older adult" in issue for issue in safety_issues):
            relaxed_target = "< 7.5–8.0%"
            turns.append((
                "Safety Agent",
                f"Because of age and hypoglycaemia risk, I recommend relaxing the HbA1c target to {relaxed_target}.",
            ))

        if relaxed_target != target_initial:
            turns.append((
                "Guideline Agent",
                f"I accept this safety constraint and can work with a target of {relaxed_target}, while still aiming to improve control.",
            ))
        final_target = relaxed_target
    else:
        turns.append((
            "Safety Agent",
            "I do not identify major additional safety concerns with this simplified information.",
        ))
        final_target = target_initial

    turns.append((
        "Coordinator",
        f"For teaching purposes, we document the agreed illustrative HbA1c target as {final_target}. In practice, the clinician and patient would decide together.",
    ))

    return turns, final_target, target_initial


def main():
    st.set_page_config(
        page_title="Multi-Agent Demo: Virtual Diabetes Case Conference",
        layout="wide",
    )

    st.title("Multi-Agent Framework Demo: Virtual Diabetes Case Conference")
    st.markdown(
        """
This app demonstrates a **multi-agent framework** for a simplified diabetes case.
It is designed for **teaching and academic discussion**, not for real clinical decision-making.

Use the tabs below to either:
- **Run the interactive multi-agent demo** on a single patient case, or
- **Study different multi-agent system (MAS) architectures** that could be used for this scenario.
        """
    )

    tab_demo, tab_arch, tab_types = st.tabs([
        "Interactive demo",
        "MAS architectures (theory)",
        "Agent types (paradigms)",
    ])

    # --- Tab 1: existing interactive demo ---
    with tab_demo:
        st.sidebar.markdown("### 0. Choose patient source")
        source = st.sidebar.radio(
            "Patient source",
            ["Manual entry", "Synthetic dataset"],
            index=0,
        )

        if source == "Manual entry":
            case = build_patient_case()
        else:
            df = load_synthetic_diabetes()
            st.sidebar.markdown("### Synthetic dataset selection")
            idx = st.sidebar.selectbox(
                "Select synthetic patient (row)",
                options=list(df.index),
                format_func=lambda i: (
                    f"Row {i} – Age "
                    f"{('NA' if pd.isna(df.loc[i, 'PatientAge']) else int(df.loc[i, 'PatientAge']))}, "
                    f"BG {('NA' if pd.isna(df.loc[i, 'BloodGlucose']) else df.loc[i, 'BloodGlucose'])} mg/dL, "
                    f"DiabetesStatus {('NA' if pd.isna(df.loc[i, 'DiabetesStatus']) else int(df.loc[i, 'DiabetesStatus']))}"
                ),
            )
            case = patient_case_from_row(df.loc[idx])

        col1, col2 = st.columns([2, 3])

        with col1:
            st.subheader("1. Patient snapshot (for students)")
            st.write(
                f"{case.age}-year-old {case.sex.lower()} with type 2 diabetes for "
                f"{case.duration_dm_years:.1f} years, BMI {case.bmi:.1f} kg/m²."
            )
            st.write(f"HbA1c: {case.hba1c:.1f}%  |  eGFR: {case.eGFR:.0f} mL/min")
            st.write(
                f"Therapy: "
                f"{'Metformin, ' if case.on_metformin else ''}"
                f"{'Sulfonylurea, ' if case.on_sulfonylurea else ''}"
                f"{'SGLT2 inhibitor, ' if case.on_sglt2 else ''}".rstrip(", ")
                or "No recorded medications."
            )
            st.write(
                f"Comorbidities: "
                f"{'CVD, ' if case.has_cvd else ''}"
                f"{'CKD, ' if case.has_ckd else ''}"
                f"{'Hypoglycaemia history, ' if case.hypoglycemia_history else ''}".rstrip(", ")
                or "No recorded comorbidities."
            )
            st.markdown(f"**Patient's main concern**: {case.main_concern}")

            st.info(
                "This is a **didactic tool**. The logic is intentionally simplified and must not be used "
                "for real clinical decisions."
            )

            run_button = st.button("Run multi-agent case conference")

        if 'agent_outputs' not in st.session_state:
            st.session_state.agent_outputs = None

        if run_button:
            st.session_state.agent_outputs = run_multi_agent_pipeline(case)

        with col2:
            st.subheader("2. Agent reasoning steps")

            if st.session_state.agent_outputs is None:
                st.write("Click **'Run multi-agent case conference'** to see how the agents collaborate.")
            else:
                # Identify workflow output (if present) for use in the Coordinator box
                workflow_out = None
                for ao in st.session_state.agent_outputs:
                    if ao.name == "Workflow / Care Coordination Agent":
                        workflow_out = ao
                        break

                for ao in st.session_state.agent_outputs:
                    with st.expander(f"{ao.name}", expanded=(ao.name != "Coordinator Agent")):
                        st.markdown(f"**Role:** {ao.role}")
                        st.markdown("**Internal reasoning (for teaching):**")
                        st.code(ao.reasoning, language="markdown")
                        st.markdown("**Output to other agents / clinician:**")
                        st.markdown(ao.message)

                        # Attach human-in-the-loop workflow exercise to the Coordinator Agent at the end
                        if ao.name == "Coordinator Agent" and workflow_out is not None:
                            tasks = workflow_out.structured.get("tasks", [])
                            if tasks:
                                st.markdown("---")
                                st.markdown(
                                    "**Which of these workflow steps would you *actually* carry out?**"
                                )
                                accepted = []
                                for t in tasks:
                                    if st.checkbox(t, value=True, key=f"wf_{t}"):
                                        accepted.append(t)

                                st.markdown("**You marked the following steps as accepted in this exercise:**")
                                if accepted:
                                    for t in accepted:
                                        st.markdown(f"- {t}")
                                else:
                                    st.markdown(
                                        "(No steps selected – in real care you would need some follow-up plan.)"
                                    )

            with st.expander("Guideline vs Safety: example conversation", expanded=False):
                turns, final_target, initial_target = build_guideline_safety_conversation(case)
                for speaker, message in turns:
                    st.markdown(f"**{speaker}:** {message}")
                st.markdown(
                    f"**Guideline's initial illustrative HbA1c target:** {initial_target}"\
                    f"  |  **After Safety negotiation:** {final_target}"
                )
                st.caption(
                    "This is a simplified, scripted conversation to illustrate how agents might negotiate. "
                    "It is not a real dialogue system and must not be used for clinical decisions."
                )

                # Workflow agent proposal based on the negotiated plan
                intake_tmp = IntakeTriageAgent().act(case)
                guideline_tmp = GuidelineAgent().act(case, intake_tmp)
                safety_tmp = SafetyAgent().act(case, guideline_tmp)
                workflow_agent = WorkflowAgent()
                workflow_out = workflow_agent.act(case, guideline_tmp, safety_tmp)

                st.markdown("---")
                st.markdown("**Workflow Agent: proposed care workflow (teaching only)**")
                st.markdown(workflow_out.message)

            with st.expander("XGBoost risk model: first tree (illustrative)", expanded=False):
                booster = load_xgb_risk_model()

                # Parse the first tree dump into a simple binary tree structure
                tree_dump = booster.get_dump()[0]
                lines = [ln for ln in tree_dump.splitlines() if ln.strip()]

                nodes: Dict[int, Dict[str, Any]] = {}
                for ln in lines:
                    stripped = ln.lstrip("\t")
                    nid_str, rest = stripped.split(":", 1)
                    nid = int(nid_str)

                    if "leaf=" in rest:
                        leaf_val = rest.split("leaf=")[1].strip()
                        nodes[nid] = {"id": nid, "label": f"leaf = {float(leaf_val):.2f}"}
                    else:
                        # Example: 0:[f2<8.46] yes=1,no=2,missing=2
                        inside = rest[rest.find("[") + 1 : rest.find("]")]
                        try:
                            yes = int(rest.split("yes=")[1].split(",")[0])
                            no = int(rest.split("no=")[1].split(",")[0])
                        except Exception:
                            yes, no = None, None
                        nodes[nid] = {
                            "id": nid,
                            "label": inside,
                            "yes": yes,
                            "no": no,
                        }

                # Derive depths by traversing from the root
                depths: Dict[int, int] = {0: 0}
                stack = [0]
                while stack:
                    nid = stack.pop()
                    depth = depths.get(nid, 0)
                    node = nodes.get(nid, {})
                    for child_key in ("yes", "no"):
                        cid = node.get(child_key)
                        if cid is not None and cid in nodes and cid not in depths:
                            depths[cid] = depth + 1
                            stack.append(cid)

                # In-order layout for binary tree
                positions: Dict[int, Any] = {}
                current_x = 0

                def inorder(nid: int):
                    nonlocal current_x
                    node = nodes.get(nid)
                    if not node:
                        return
                    left = node.get("yes")
                    right = node.get("no")
                    if left is not None:
                        inorder(left)
                    positions[nid] = (current_x, -depths.get(nid, 0))
                    current_x += 1
                    if right is not None:
                        inorder(right)

                inorder(0)

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.axis("off")

                # Draw edges
                for nid, node in nodes.items():
                    x, y = positions.get(nid, (0, 0))
                    for child_key, color in (("yes", "green"), ("no", "red")):
                        cid = node.get(child_key)
                        if cid is not None and cid in positions:
                            cx, cy = positions[cid]
                            ax.plot([x, cx], [y, cy], color="0.7")
                            mid_x, mid_y = (x + cx) / 2, (y + cy) / 2
                            ax.text(mid_x, mid_y, child_key, fontsize=7, color=color)

                # Draw nodes
                for nid, node in nodes.items():
                    x, y = positions.get(nid, (0, 0))
                    ax.scatter([x], [y], s=200, color="white", edgecolors="black")
                    ax.text(
                        x,
                        y,
                        node.get("label", ""),
                        ha="center",
                        va="center",
                        fontsize=7,
                        wrap=True,
                    )

                ax.set_ylim(min(y for _, y in positions.values()) - 1, 1)
                ax.set_xlim(min(x for x, _ in positions.values()) - 1, max(x for x, _ in positions.values()) + 1)

                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)

                st.caption(
                    "Custom binary tree visualisation of the first XGBoost tree. Green edges are 'yes' branches, "
                    "red edges are 'no' branches. This is trained on synthetic data and is for teaching only, not "
                    "for clinical use."
                )

                st.markdown(
                    "**Feature legend for this synthetic model:**  "
                    "`f0` = age (years),  `f1` = BMI (kg/m²),  `f2` = HbA1c (%),  `f3` = eGFR (mL/min).  "
                    "So, for example, a node labelled `f2<8.4` means 'HbA1c below 8.4%?'."
                )

                st.markdown("---")
                st.markdown("**Human-in-the-loop teaching exercise**")
                choice = st.radio(
                    "As the clinician, which HbA1c strategy would you choose (for teaching only)?",
                    [
                        f"Option 1: Guideline-focused (keep initial target {initial_target})",
                        f"Option 2: Safety-focused (use negotiated target {final_target})",
                        "I am unsure / would need more information",
                    ],
                    index=1,
                )

                clinician_target = st.text_input(
                    "Clinician's final documented HbA1c target (you may adjust the text):",
                    value=final_target,
                )

                if choice.startswith("Option 1"):
                    st.info(
                        "You chose the more guideline-focused option. In practice, you would still need to "
                        "check patient preferences, comorbidities, and feasibility before implementing it."
                    )
                elif choice.startswith("Option 2"):
                    st.info(
                        "You chose the more safety-focused option. This often aligns with recommendations for "
                        "frail or older adults, but shared decision-making with the patient remains essential."
                    )
                else:
                    st.info(
                        "Being unsure and seeking more information (or a second opinion) is a valid and often "
                        "safe choice in complex clinical situations."
                    )

                st.markdown(
                    f"**Clinician's final (teaching-only) target recorded here:** {clinician_target}"
                )

    # --- Tab 2: MAS architectures for this case ---
    with tab_arch:
        st.subheader("How could different multi-agent system architectures look in this case?")

        st.markdown(
            """
We consider the same set of agents you see in the interactive demo:

- **Intake & Triage Agent** – frames the case and red flags
- **Guideline Agent** – applies simplified diabetes guidelines
- **Safety Agent** – screens for contraindications and risks
- **Patient Education Agent** – explains the plan in lay language
- **Coordinator Agent** – orchestrates and summarises

Below are different ways to organise these agents as a *multi-agent system (MAS)*.
            """
        )

        st.markdown("### 1. Centralised coordinator (what this app roughly uses)")
        st.markdown(
            """
In this design, a **Coordinator Agent** acts as a conductor:

```text
          +-------------------+
          |   Clinician / UI  |
          +----------+--------+
                     |
                     v
          +----------+--------+
          |  Coordinator      |
          +----------+--------+
             |        |       \
             v        v        v
     +-------+--+  +--+-------+--+  +-----------------+
     | Intake  |  | Guideline |  | Safety           |
     +---------+  +-----------+  +-----------------+
                              \
                               v
                        +------+------+
                        | Patient Ed. |
                        +-------------+
```

- Data flows **down** from the coordinator to specialist agents.
- Outputs flow **back up** to the coordinator, which builds a summary.

This matches the behaviour in the *Interactive demo* tab.
            """
        )

        st.markdown("### 2. Blackboard / shared memory MAS")
        st.markdown(
            """
Instead of calling each other directly, agents communicate through a shared **blackboard**:

```text
         +----------------------+
         |   Shared blackboard  |
         |  (patient case,      |
         |   summaries, plans)  |
         +----------+-----------+
                    ^
      --------------+-----------------------------
      |            |            |               |
      v            v            v               v
  +--------+   +--------+   +--------+     +-----------+
  | Intake |   | Guide. |   | Safety |     | PatientEd |
  +--------+   +--------+   +--------+     +-----------+
```

- Each agent **reads** from and **writes** to the blackboard.
- For example:
  - Intake writes `intake_summary`, `red_flags`.
  - Guideline reads those and writes `guideline_plan`.
  - Safety reads plan and writes `safety_warnings`.
  - Patient Education reads everything and writes `patient_text`.

This design emphasises a **shared knowledge space** and loose coupling between agents.
            """
        )

        st.markdown("### 3. Peer-to-peer MAS with negotiation")
        st.markdown(
            r"""
Here, agents can **communicate directly** and even **disagree**:

```text
  Intake  →  Guideline  ↔  Safety
                   \\       /
                    v     v
                 Patient Ed.
```

Example for this diabetes case:

- Guideline Agent: *"Propose tight control, add GLP‑1 RA."*
- Safety Agent: *"Objection: older patient with hypoglycaemia history, relax target."*
- They exchange messages until they reach a compromise or present
  **multiple options** to the clinician.

This is useful to teach that MAS can represent **different specialist opinions**
rather than a single pipeline.
            """
        )

        st.markdown("### 4. Hierarchical and human-in-the-loop MAS")
        st.markdown(
            """
Finally, you can place this patient-level MAS inside a **hierarchy** and keep
the **clinician explicitly in the loop**:

```text
   Population-level agents (clinic / population health)
                 ^
                 |
      Summaries from many patient-level MAS
                 ^
                 |
        Patient-level multi-agent case conference
                 ^
                 |
             Clinician
```

- Many patient-level case conferences run in parallel (one per patient).
- Their outcomes feed population-level agents (e.g. resource planning, quality & safety).
- The **clinician remains the final decision-maker**, using agents as structured
  decision support rather than autonomous controllers.

This is closer to how AI support tools should be used in real healthcare settings.
            """
        )

    # --- Tab 3: Agent types and paradigms ---
    with tab_types:
        st.subheader("Agent types in this demo (paradigms)")
        st.markdown(
            """
We can also classify the agents by **type of behaviour**:

- **Reactive agent** – responds to signals and thresholds
- **Deliberative / reasoning agent** – plans using models or rules
- **Interface / communication agent** – focuses on explanation and interaction
- **Coordinator / organisational agent** – manages other agents

In this diabetes case conference:

- Intake & Triage Agent → mostly *deliberative* framing of the case
- Guideline Agent → *deliberative* (rule-based guideline reasoning)
- Safety Agent → largely *reactive* (threshold checks, risk flags)
- Patient Education Agent → *interface* agent (patient communication)
- Coordinator Agent → *organisational* / coordinator agent

You can relate this to common AI agent models such as **BDI (Belief–Desire–Intention)**:

- *Beliefs*: patient data, summaries, and plans shared between agents
- *Desires*: good glycaemic control, safety, understandable explanation
- *Intentions*: concrete therapy suggestions and safety recommendations

This tab is meant to help students link the concrete demo to more abstract
notions of **agent types** and **agent architectures**.
            """
        )


if __name__ == "__main__":
    main()
