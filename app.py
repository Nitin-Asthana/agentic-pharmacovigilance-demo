"""
Agentic Pharmacovigilance Case Processing — Leadership Demonstration
===================================================================

Two pages:
  1) Case Command Center — a team of specialized AI agents work ONE safety
     signal through a shared case state, with a human PV reviewer in control.
  2) Enterprise PV Dashboard — synthetic portfolio view (case counts, status,
     routes) so leadership sees the operational picture.

Genuine agent behaviour:
  - Intake Agent      (Azure OpenAI call #1)  -> classifies the signal
  - Extraction Agent  (Azure OpenAI call #2)  -> structures the case (uses Intake output)
  - History Agent     (rules)                 -> matches similar prior cases
  - Validation Agent  (rules)                 -> completeness / gaps
  - Triage Agent      (rules)                 -> route / priority
  - Human PV Reviewer (you)                   -> final decision

Run:  az login  ->  streamlit run app.py
Auth: Microsoft Entra ID via DefaultAzureCredential (no API keys).
Env:  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION
Fallback: deterministic Simulation Mode with identical UI. All data synthetic.
"""

import html
import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import streamlit as st
from pydantic import BaseModel, Field

try:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from openai import AzureOpenAI

    _AZURE_SDK = True
except Exception:  # pragma: no cover
    _AZURE_SDK = False


# ===========================================================================
# HELPERS
# ===========================================================================
def H(s: str) -> str:
    return "\n".join(ln.strip() for ln in s.splitlines() if ln.strip())


def esc(x) -> str:
    return html.escape(str(x)) if x is not None else ""


# ===========================================================================
# SHARED CASE STATE
# ===========================================================================
class PVCaseState(BaseModel):
    case_id: str
    channel: str
    raw_message: str
    signal_detected: bool = False
    signal_type: Optional[str] = None
    initial_priority: Optional[str] = None
    intake_reason: Optional[str] = None
    patient: Dict[str, Optional[str]] = Field(default_factory=dict)
    reporter: Dict[str, Optional[str]] = Field(default_factory=dict)
    product: Dict[str, Optional[str]] = Field(default_factory=dict)
    adverse_events: List[str] = Field(default_factory=list)
    onset: Optional[str] = "Not reported"
    outcome: Optional[str] = "Not reported"
    hospitalization: Optional[str] = "Unknown"
    seriousness: Optional[str] = None
    category: Optional[str] = None
    confidence: Optional[int] = None
    evidence: List[str] = Field(default_factory=list)
    similar_cases: List[Dict[str, str]] = Field(default_factory=list)
    validation_checks: List[Dict[str, str]] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    validation_status: str = "Pending"
    recommended_route: Optional[str] = None
    triage_priority: Optional[str] = None
    triage_reason: Optional[str] = None
    next_steps: List[str] = Field(default_factory=list)
    human_decision: Optional[str] = None
    case_status: str = "New"
    closed: bool = False
    disposition: Optional[str] = None
    followup_text: Optional[str] = None
    actions_done: List[str] = Field(default_factory=list)
    agent_log: List[Dict[str, str]] = Field(default_factory=list)


# ===========================================================================
# SYNTHETIC SCENARIOS + HISTORICAL DATA (all fictional)
# ===========================================================================
CHANNELS = ["Patient WhatsApp", "HCP Email", "Call Center Call", "Social Media Post"]
CHANNEL_ICON = {"Patient WhatsApp": "\U0001F4AC", "HCP Email": "\u2709\uFE0F",
                "Call Center Call": "\U0001F4DE", "Social Media Post": "\U0001F310"}
SOURCE_BY_CHANNEL = {"Patient WhatsApp": "Patient", "HCP Email": "HCP",
                     "Call Center Call": "Call Center", "Social Media Post": "Social Media"}
PRODUCTS = {"cardvexa": "Cardvexa", "neuroxil": "Neuroxil", "respivance": "Respivance"}

# Per-channel sample messages so changing the Channel also updates the message.
CHANNEL_SAMPLES = {
    "Patient WhatsApp": ("Hi, I started Cardvexa 500 mg three days ago. Since yesterday I've developed a "
                         "severe skin rash and difficulty breathing. I was taken to the hospital last night."),
    "HCP Email": ("From: Dr. A. Sharma <a.sharma@democlinic.example>\nSubject: Suspected ADR \u2014 Neuroxil 10 mg\n\n"
                  "A 56-year-old female patient started Neuroxil 10 mg five days ago and developed persistent "
                  "vomiting and dizziness, treated in the emergency department."),
    "Call Center Call": ("Agent: How can I help?\nCaller: My father started Respivance 250 mg last week and has had "
                         "mild nausea and a headache since. He has not been hospitalised."),
    "Social Media Post": ("Has anyone else experienced severe itching after taking Cardvexa? I started it last week "
                          "and noticed the symptoms yesterday. #sideeffects"),
}

# One distinct message for every Scenario x Channel combination.
MESSAGES = {
    ("Serious Adverse Event", "Patient WhatsApp"):
        ("Hi, I'm a 58-year-old man. Three days ago I started Cardvexa 500 mg for my blood pressure. Since yesterday "
         "I've had a severe skin rash spreading across my chest and real difficulty breathing. My wife took me to "
         "hospital last night and I was kept in overnight. I'm home now and slowly recovering, but I'm quite worried "
         "and wanted to report what happened."),
    ("Serious Adverse Event", "HCP Email"):
        ("From: Dr. A. Sharma <a.sharma@democlinic.example>\nTo: drugsafety@demopharma.example\n"
         "Subject: Suspected serious ADR, Neuroxil 10 mg\n\nDear Drug Safety Team,\n\n"
         "A 64-year-old female patient commenced Neuroxil 10 mg five days ago for neuropathic pain. Yesterday she "
         "developed severe facial swelling and difficulty breathing and was admitted to the emergency department "
         "overnight. She has since recovered. Kindly advise on expedited reporting.\n\nRegards,\nDr. A. Sharma"),
    ("Serious Adverse Event", "Call Center Call"):
        ("Agent: Thank you for calling the safety line, how can I help?\nCaller: I'm calling about my father, a "
         "71-year-old man. He started Respivance 250 mg about three days ago and yesterday he collapsed with severe "
         "dizziness and vomiting. We rushed him to hospital and he was admitted overnight. He is recovering now but "
         "it was frightening.\nAgent: I'm sorry to hear that. I'll log this as a serious report."),
    ("Serious Adverse Event", "Social Media Post"):
        ("I'm a 46-year-old woman and I want to warn people. Started Cardvexa 500 mg last week and got a terrible "
         "skin rash all over plus difficulty breathing. Ended up in hospital overnight. Thankfully I have since "
         "recovered. Has anyone else reacted like this? #sideeffects #Cardvexa"),

    ("Non-Serious Adverse Event", "Patient WhatsApp"):
        ("Hey, I've been on Respivance 250 mg for about a week now for my reflux. I keep getting mild nausea and a "
         "dull headache in the afternoons. It's annoying but not too bad and I've not needed to see a doctor. Is this "
         "a known side effect?"),
    ("Non-Serious Adverse Event", "HCP Email"):
        ("From: Dr. R. Mehta <r.mehta@democlinic.example>\nSubject: Non-serious ADR, Neuroxil 10 mg\n\n"
         "A patient started Neuroxil 10 mg last week and reports intermittent mild dizziness and nausea. No "
         "hospitalisation and the symptoms are tolerable. Logging for completeness; the outcome is not yet known.\n"),
    ("Non-Serious Adverse Event", "Call Center Call"):
        ("Agent: How can I help today?\nCaller: My mother started Cardvexa 500 mg last week. She's had some mild "
         "itching on her arms and the odd headache. Nothing serious, she has not been hospitalised. We just wanted "
         "to report it.\nAgent: Understood, I'll note it down."),
    ("Non-Serious Adverse Event", "Social Media Post"):
        ("Anyone else on Neuroxil 10 mg getting mild nausea and headaches? Started mine last week, nothing major, "
         "just wondering if it settles down. #Neuroxil #meds"),

    ("Product Quality Complaint", "Patient WhatsApp"):
        ("Hi, I picked up my repeat prescription of Neuroxil 10 mg yesterday and when I opened the box several "
         "tablets were cracked and a strange yellow colour. The blister foil was also torn. I haven't taken any. "
         "Can I get a replacement?"),
    ("Product Quality Complaint", "HCP Email"):
        ("From: Pharmacy <pharmacy@democlinic.example>\nSubject: Product quality complaint, Cardvexa 500 mg\n\n"
         "A patient returned a pack of Cardvexa 500 mg containing discoloured, chipped tablets and a damaged "
         "blister. No adverse event reported. Please advise on the returns process.\n"),
    ("Product Quality Complaint", "Call Center Call"):
        ("Agent: How can I help?\nCaller: I bought a box of Respivance 250 mg and half the tablets are broken and "
         "discoloured, and the blister pack looks crushed. I've not taken any of them. What should I do?\n"
         "Agent: Thanks, I'll raise a quality complaint."),
    ("Product Quality Complaint", "Social Media Post"):
        ("Just opened a fresh box of Cardvexa 500 mg and the tablets are cracked and weirdly discoloured, blister "
         "all damaged. Not risking taking these. Anyone had dodgy packaging? #Cardvexa #qualityissue"),

    ("Medical Information Request", "Patient WhatsApp"):
        ("Hi, quick question. Is it OK to take Respivance 250 mg together with an over-the-counter antacid, or should "
         "I space them out? No side effects, I just want to get the timing right. Thanks!"),
    ("Medical Information Request", "HCP Email"):
        ("From: Dr. R. Mehta <r.mehta@democlinic.example>\nSubject: Dosing query, Neuroxil 10 mg\n\n"
         "Could you confirm whether Neuroxil 10 mg can be co-administered with a common antacid, and clarify the "
         "correct dosing schedule for elderly patients? No adverse events to report.\n"),
    ("Medical Information Request", "Call Center Call"):
        ("Agent: How can I help?\nCaller: I just have a question about Cardvexa 500 mg. Can I take it with my usual "
         "antacid, and what's the best time of day for the dosing? No problems, just checking.\n"
         "Agent: Sure, I'll pass this to medical information."),
    ("Medical Information Request", "Social Media Post"):
        ("Does anyone know if you can take Respivance 250 mg together with an antacid? Just trying to sort out the "
         "dosing timing, no issues otherwise. #Respivance #question"),

    ("Incomplete Case", "Patient WhatsApp"):
        ("Hi, a friend of mine mentioned she felt a bit unwell after starting one of your medicines recently, but "
         "she didn't say which one or what exactly happened. Just flagging it in case it matters."),
    ("Incomplete Case", "HCP Email"):
        ("From: Reception <reception@democlinic.example>\nSubject: Possible issue reported\n\n"
         "A caller left a message saying a patient felt unwell after beginning one of the products, but gave no "
         "name, product or symptoms. Passing this on for your awareness.\n"),
    ("Incomplete Case", "Call Center Call"):
        ("Agent: How can I help?\nCaller: My neighbour said he hasn't felt right since starting some new tablets "
         "recently. I think it was one of yours but I honestly don't know the name or the details.\n"
         "Agent: No problem, I'll log what we have."),
    ("Incomplete Case", "Social Media Post"):
        ("Heard from someone that they felt off after starting a new medicine recently, couldn't tell me the brand "
         "or what happened though. Anyone know more? #unsure"),
}

SCENARIOS = {
    "Serious Adverse Event": {
        "channel": "Patient WhatsApp",
        "message": ("Hi, I started Cardvexa 500 mg three days ago. Since yesterday I've developed a "
                    "severe skin rash and difficulty breathing. I was taken to the hospital last night."),
    },
    "Non-Serious Adverse Event": {
        "channel": "Call Center Call",
        "message": ("Agent: How can I help?\nCaller: My father started Respivance 250 mg last week and "
                    "has had mild nausea and a headache since. He has not been hospitalised."),
    },
    "Product Quality Complaint": {
        "channel": "Patient WhatsApp",
        "message": ("I opened a new box of Neuroxil 10 mg and several tablets were cracked and discolored. "
                    "The blister pack also looked damaged. I have not taken any of the affected tablets."),
    },
    "Medical Information Request": {
        "channel": "HCP Email",
        "message": ("From: Dr. R. Mehta <r.mehta@democlinic.example>\nSubject: Dosing query — Respivance 250 mg\n\n"
                    "Could you confirm whether Respivance 250 mg can be taken with a common antacid, and the "
                    "correct dosing schedule? No adverse events to report."),
    },
    "Incomplete Case": {
        "channel": "Social Media Post",
        "message": ("Someone said they felt unwell after starting one of the products recently. "
                    "No other details were provided."),
    },
}

# Synthetic historical case database (portfolio seed for dashboard + matching)
def _seed_history() -> List[Dict[str, str]]:
    rows = [
        ("Cardvexa", "Rash", "Adverse Event", "Serious", "Safety Review", "Recovered"),
        ("Cardvexa", "Dyspnoea", "Adverse Event", "Serious", "Safety Review", "Recovering"),
        ("Cardvexa", "Pruritus", "Adverse Event", "Non-Serious", "Safety Review", "Recovered"),
        ("Cardvexa", "Rash", "Adverse Event", "Non-Serious", "Safety Review", "Recovered"),
        ("Neuroxil", "Dizziness", "Adverse Event", "Non-Serious", "Safety Review", "Recovered"),
        ("Neuroxil", "Vomiting", "Adverse Event", "Serious", "Safety Review", "Recovering"),
        ("Neuroxil", "Cracked tablets", "Product Quality Complaint", "Non-Serious", "Quality Review", "N/A"),
        ("Neuroxil", "Discolored tablets", "Product Quality Complaint", "Non-Serious", "Quality Review", "N/A"),
        ("Respivance", "Nausea", "Adverse Event", "Non-Serious", "Safety Review", "Recovered"),
        ("Respivance", "Headache", "Adverse Event", "Non-Serious", "Safety Review", "Recovered"),
        ("Respivance", "Dosing question", "Medical Information Request", "Unknown", "Medical Information", "N/A"),
        ("Respivance", "Dizziness", "Adverse Event", "Non-Serious", "Follow-up Required", "Unknown"),
        ("Cardvexa", "Angioedema", "Adverse Event", "Serious", "Safety Review", "Recovering"),
        ("Cardvexa", "Rash", "Adverse Event", "Non-Serious", "Safety Review", "Recovered"),
        ("Neuroxil", "Nausea", "Adverse Event", "Non-Serious", "Safety Review", "Recovered"),
        ("Respivance", "Fatigue", "Adverse Event", "Non-Serious", "Follow-up Required", "Unknown"),
        ("Cardvexa", "Dyspnoea", "Adverse Event", "Serious", "Safety Review", "Fatal"),
        ("Neuroxil", "Packaging defect", "Product Quality Complaint", "Non-Serious", "Quality Review", "N/A"),
        ("Respivance", "Dosing question", "Medical Information Request", "Unknown", "Medical Information", "N/A"),
        ("Cardvexa", "Pruritus", "Adverse Event", "Non-Serious", "Safety Review", "Recovered"),
    ]
    out = []
    base = datetime.now()
    for i, (prod, ev, cat, ser, route, outc) in enumerate(rows):
        out.append({
            "case_id": f"HIST-{2026}-{i+1:04d}",
            "date": (base - timedelta(days=6 * (i + 2))).strftime("%d %b %Y"),
            "product": prod, "event": ev, "category": cat, "seriousness": ser,
            "route": route, "outcome": outc, "source": "Historical",
        })
    return out


HISTORY = _seed_history()

JOURNEY = ["Signal", "AI Understand", "Validate", "Triage", "PV Review", "Next Step"]


# ===========================================================================
# AZURE OPENAI
# ===========================================================================
def azure_config() -> dict:
    return {"endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", "").strip(),
            "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip(),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview").strip()}


@st.cache_resource(show_spinner=False)
def get_azure_client():
    if not _AZURE_SDK:
        return None
    cfg = azure_config()
    if not cfg["endpoint"] or not cfg["deployment"]:
        return None
    try:
        tp = get_bearer_token_provider(DefaultAzureCredential(),
                                       "https://cognitiveservices.azure.com/.default")
        client = AzureOpenAI(azure_endpoint=cfg["endpoint"], azure_ad_token_provider=tp,
                             api_version=cfg["api_version"], timeout=30.0, max_retries=1)
        return {"client": client, "deployment": cfg["deployment"]}
    except Exception:
        return None


def _azure_json(azure, system: str, user: str, schema: dict, name: str) -> dict:
    kwargs = dict(
        model=azure["deployment"],
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": name, "strict": True, "schema": schema}},
    )
    try:
        r = azure["client"].chat.completions.create(temperature=0, **kwargs)
    except Exception as exc:  # noqa: BLE001
        if "temperature" in str(exc).lower() or "unsupported" in str(exc).lower():
            r = azure["client"].chat.completions.create(**kwargs)
        else:
            raise
    return json.loads(r.choices[0].message.content)


# ---- Agent 1: Intake ------------------------------------------------------
INTAKE_SYSTEM = """You are the INTAKE AGENT in a pharmacovigilance AI team (decision support only).
Read the incoming communication and decide whether it may contain a PV-relevant signal.
Use only facts present in the text. No advice, no hidden reasoning.
Return: signal_detected (bool); signal_type ("Potential Adverse Event" | "Product Quality Complaint" |
"Medical Information Request" | "No PV Signal"); initial_priority ("High" | "Medium" | "Low");
intake_reason (one concise sentence)."""

INTAKE_SCHEMA = {
    "type": "object",
    "properties": {
        "signal_detected": {"type": "boolean"},
        "signal_type": {"type": "string", "enum": ["Potential Adverse Event", "Product Quality Complaint",
                                                   "Medical Information Request", "No PV Signal"]},
        "initial_priority": {"type": "string", "enum": ["High", "Medium", "Low"]},
        "intake_reason": {"type": "string"},
    },
    "required": ["signal_detected", "signal_type", "initial_priority", "intake_reason"],
    "additionalProperties": False,
}

# ---- Agent 2: Extraction --------------------------------------------------
EXTRACTION_SYSTEM = """You are the PV EXTRACTION AGENT in a pharmacovigilance AI team (decision support only).
The Intake Agent already classified this communication. Extract the structured case using ONLY facts in the text.
Use "Unknown"/"Not reported"/null when absent. No advice, no hidden reasoning. evidence = short verbatim quotes.
Populate patient{age,gender,patient_id}, reporter{reporter_type}, product{name,strength,dose,route},
adverse_events, onset, outcome, hospitalization ("Yes"/"No"/"Unknown"),
seriousness ("Serious"/"Non-Serious"/"Unknown" — treat hospitalization, airway compromise, ICU,
life-threatening reactions or death as Serious), category, confidence (0-100)."""

_STRN = {"type": ["string", "null"]}
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "patient": {"type": "object",
                    "properties": {"age": {"type": "string"}, "gender": {"type": "string"}, "patient_id": _STRN},
                    "required": ["age", "gender", "patient_id"], "additionalProperties": False},
        "reporter": {"type": "object", "properties": {"reporter_type": {"type": "string"}},
                     "required": ["reporter_type"], "additionalProperties": False},
        "product": {"type": "object",
                    "properties": {"name": _STRN, "strength": _STRN, "dose": _STRN, "route": _STRN},
                    "required": ["name", "strength", "dose", "route"], "additionalProperties": False},
        "adverse_events": {"type": "array", "items": {"type": "string"}},
        "onset": {"type": "string"}, "outcome": {"type": "string"}, "hospitalization": {"type": "string"},
        "seriousness": {"type": "string", "enum": ["Serious", "Non-Serious", "Unknown"]},
        "category": {"type": "string",
                     "enum": ["Adverse Event", "Product Quality Complaint", "Medical Information Request"]},
        "confidence": {"type": "integer"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["patient", "reporter", "product", "adverse_events", "onset", "outcome", "hospitalization",
                 "seriousness", "category", "confidence", "evidence"],
    "additionalProperties": False,
}


# ===========================================================================
# SIMULATION (deterministic) — split to mirror the two LLM agents
# ===========================================================================
def _mock_full(message: str, channel: str) -> dict:
    t = message.lower()
    product = next((n for k, n in PRODUCTS.items() if k in t), None)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml)\b", t)
    strength = f"{m.group(1)} {m.group(2)}" if m else None
    terms = {"skin rash": "Skin rash", "difficulty breathing": "Difficulty breathing", "itching": "Itching",
             "vomiting": "Vomiting", "nausea": "Nausea", "headache": "Headache", "dizziness": "Dizziness"}
    events = [lbl for key, lbl in terms.items() if key in t]
    evidence = [f'"{k}"' for k in terms if k in t][:5]
    quality = any(k in t for k in ("cracked", "discolored", "damaged", "blister")) and "difficulty breathing" not in t
    medinfo = any(k in t for k in ("confirm whether", "dosing schedule", "taken with", "correct dosing",
                                   "can i take", "is it ok to take", "together with", "co-administered",
                                   "with a common antacid", "with an antacid", "with my usual antacid",
                                   "dosing", "space them out"))
    if quality:
        category, signal_type, events = "Product Quality Complaint", "Product Quality Complaint", []
    elif medinfo:
        category, signal_type, events = "Medical Information Request", "Medical Information Request", []
    else:
        category, signal_type = "Adverse Event", "Potential Adverse Event"
    neg_hosp = any(k in t for k in ("wasn't hospitalised", "was not hospitalised", "not hospitalised",
                                    "no hospital", "not been hospitalised", "not admitted"))
    if neg_hosp:
        hosp = "No"
    elif any(k in t for k in ("admitted", "emergency", "intensive care", "icu")) or "hospital" in t:
        hosp = "Yes"
    else:
        hosp = "Unknown"
    if category != "Adverse Event":
        seriousness = "Non-Serious" if category == "Product Quality Complaint" else "Unknown"
    elif hosp == "Yes" or "difficulty breathing" in t:
        seriousness = "Serious"
    elif events:
        seriousness = "Non-Serious"
    else:
        seriousness = "Unknown"
    age, gender = "Unknown", "Unknown"
    am = re.search(r"(\d{1,3})[- ]year[- ]old", t)
    if am:
        age = am.group(1)
    if "female" in t or "woman" in t or " she " in t or "mother" in t:
        gender = "Female"
    elif "male" in t or re.search(r"\bman\b", t) or "father" in t or " he " in t or " his " in t:
        gender = "Male"
    onset = "Not reported"
    for p in ("three days ago", "five days ago", "last week", "yesterday"):
        if p in t:
            onset = p.capitalize()
            break
    outcome = "Not reported"
    if any(k in t for k in ("recovered", "fully resolved", "since resolved", "have resolved",
                            "has resolved", "all resolved", "resolved now")):
        outcome = "Recovered"
    elif "recovering" in t or "resolving" in t:
        outcome = "Recovering"
    elif "ongoing" in t or "not resolved" in t:
        outcome = "Ongoing"
    elif any(k in t for k in ("died", "fatal", "passed away")):
        outcome = "Fatal"
    signal_detected = signal_type != "No PV Signal"
    priority = "High" if seriousness == "Serious" else ("Low" if category == "Medical Information Request" else "Medium")
    reason = {"Adverse Event": "Reporter describes symptoms following product use"
                               + (" with hospitalization" if hosp == "Yes" else "") + ".",
              "Product Quality Complaint": "Reporter describes a product quality defect; no adverse event stated.",
              "Medical Information Request": "Reporter asks a product question; no adverse event stated."}[category]
    completeness = 40 + (15 if product else 0) + (15 if events else 0) + (10 if age != "Unknown" else 0) \
        + (10 if hosp != "Unknown" else 0) + (10 if onset != "Not reported" else 0)
    confidence = min(96, completeness if (events or category != "Adverse Event") else 42)
    return {"signal_detected": signal_detected, "signal_type": signal_type, "initial_priority": priority,
            "intake_reason": reason,
            "patient": {"age": age, "gender": gender, "patient_id": None},
            "reporter": {"reporter_type": SOURCE_BY_CHANNEL.get(channel, "Unknown")},
            "product": {"name": product, "strength": strength, "dose": None, "route": None},
            "adverse_events": events, "onset": onset, "outcome": outcome, "hospitalization": hosp,
            "seriousness": seriousness, "category": category, "confidence": confidence, "evidence": evidence}


# ===========================================================================
# AGENTS
# ===========================================================================
def intake_agent(message, channel, azure, connected):
    if connected:
        try:
            user = f"Channel: {channel}\n\nCommunication:\n\"\"\"\n{message}\n\"\"\""
            return _azure_json(azure, INTAKE_SYSTEM, user, INTAKE_SCHEMA, "intake"), "azure", None
        except Exception as exc:  # noqa: BLE001
            f = _mock_full(message, channel)
            return {k: f[k] for k in ("signal_detected", "signal_type", "initial_priority", "intake_reason")}, \
                "sim", f"Intake agent fell back: {exc}"
    f = _mock_full(message, channel)
    return {k: f[k] for k in ("signal_detected", "signal_type", "initial_priority", "intake_reason")}, \
        "sim", "Azure not configured."


def extraction_agent(message, channel, intake, azure, connected):
    if connected:
        try:
            user = (f"Channel: {channel}\nIntake Agent classification: {intake.get('signal_type')} "
                    f"(priority {intake.get('initial_priority')}).\n\nCommunication:\n\"\"\"\n{message}\n\"\"\"")
            return _azure_json(azure, EXTRACTION_SYSTEM, user, EXTRACTION_SCHEMA, "extraction"), "azure", None
        except Exception as exc:  # noqa: BLE001
            f = _mock_full(message, channel)
            keys = ("patient", "reporter", "product", "adverse_events", "onset", "outcome",
                    "hospitalization", "seriousness", "category", "confidence", "evidence")
            return {k: f[k] for k in keys}, "sim", f"Extraction agent fell back: {exc}"
    f = _mock_full(message, channel)
    keys = ("patient", "reporter", "product", "adverse_events", "onset", "outcome",
            "hospitalization", "seriousness", "category", "confidence", "evidence")
    return {k: f[k] for k in keys}, "sim", "Azure not configured."


def apply_intake(state, ai):
    state.signal_detected = ai["signal_detected"]
    state.signal_type = ai["signal_type"]
    state.initial_priority = ai["initial_priority"]
    state.intake_reason = ai["intake_reason"]
    out = f"{ai['signal_type']} detected" if ai["signal_detected"] else "No PV-relevant signal"
    state.agent_log.append({"agent": "Intake Agent", "status": "completed", "outcome": out})


def apply_extraction(state, ex):
    for k in ("patient", "reporter", "product", "adverse_events", "onset", "outcome",
              "hospitalization", "seriousness", "category", "confidence", "evidence"):
        setattr(state, k, ex[k])
    prod = state.product.get("name")
    if state.category == "Adverse Event":
        out = f"Extracted {prod or 'product (unspecified)'} and {len(state.adverse_events)} event(s)"
    else:
        out = f"Extracted {prod or 'product (unspecified)'} — {state.category}"
    state.agent_log.append({"agent": "Extraction Agent", "status": "completed", "outcome": out})


def history_agent(state):
    prod = (state.product.get("name") or "").lower()
    ev_words = {w.lower() for e in state.adverse_events for w in e.split()}
    matches = []
    for h in HISTORY:
        if prod and h["product"].lower() == prod:
            share_event = any(w in h["event"].lower() for w in ev_words) if ev_words else False
            same_cat = h["category"] == state.category
            if share_event or same_cat:
                matches.append(h)
    matches = matches[:6]
    state.similar_cases = matches
    serious_prior = sum(1 for h in matches if h["seriousness"] == "Serious")
    if matches:
        extra = f" ({serious_prior} previously serious)" if serious_prior else ""
        out = f"{len(matches)} similar prior case(s) for {state.product.get('name')}{extra}"
        status = "attention" if serious_prior else "completed"
    else:
        out = "No similar prior cases found"
        status = "completed"
    state.agent_log.append({"agent": "History Agent", "status": status, "outcome": out})


def validation_agent(state):
    p = state.patient
    is_ae = state.category == "Adverse Event"
    patient_ok = bool(p.get("patient_id") or (p.get("age") and p.get("age") != "Unknown")
                      or (p.get("gender") and p.get("gender") != "Unknown")
                      or SOURCE_BY_CHANNEL.get(state.channel) == "Patient")
    checks = [
        ("Suspect product", "ok" if state.product.get("name") else "warn"),
        ("Adverse event", ("ok" if state.adverse_events else "warn") if is_ae else "na"),
        ("Reporter", "ok" if state.reporter.get("reporter_type") not in (None, "", "Unknown") else "warn"),
        ("Patient details", ("ok" if patient_ok else "warn") if is_ae else "na"),
        ("Seriousness information", ("ok" if state.seriousness in ("Serious", "Non-Serious") else "warn") if is_ae else "na"),
        ("Event onset date", ("ok" if state.onset not in (None, "Not reported") else "warn") if is_ae else "na"),
        ("Patient outcome", ("ok" if state.outcome not in (None, "Not reported") else "warn") if is_ae else "na"),
    ]
    state.validation_checks = [{"label": c[0], "status": c[1]} for c in checks]
    state.missing_information = [c[0] for c in checks if c[1] == "warn"]
    gaps = len(state.missing_information)
    if gaps == 0:
        state.validation_status = "Complete"
        state.agent_log.append({"agent": "Validation Agent", "status": "completed", "outcome": "Case information complete"})
    else:
        state.validation_status = "Follow-up recommended"
        state.agent_log.append({"agent": "Validation Agent", "status": "attention",
                                "outcome": f"{gaps} information gap(s) identified"})


def triage_agent(state):
    critical_missing = (state.category == "Adverse Event"
                        and (not state.product.get("name") or not state.adverse_events))
    if state.category == "Product Quality Complaint":
        route = "Quality Review"
    elif state.category == "Medical Information Request":
        route = "Medical Information"
    elif critical_missing:
        route = "Follow-up Required"
    else:
        route = "Safety Review"
    if route == "Follow-up Required":
        priority = "Medium"
    elif state.seriousness == "Serious":
        priority = "High"
    elif state.category == "Medical Information Request":
        priority = "Low"
    else:
        priority = "Medium"
    state.recommended_route = route
    state.triage_priority = priority
    if route == "Safety Review":
        state.triage_reason = ("Potential serious adverse event"
                               + (" with hospitalization reported." if state.hospitalization == "Yes" else " reported.")) \
            if state.seriousness == "Serious" else "Adverse event reported; routine safety review recommended."
        state.next_steps = ["Create an Individual Case Safety Report (ICSR) and generate an E2B(R3) electronic draft for review",
                            f"Assign to the Safety Review workqueue ({priority} priority)",
                            "Expedited reporting if serious/unlisted; otherwise include in Periodic Benefit-Risk "
                            "Evaluation Reports (PBRERs) / Periodic Safety Update Reports (PSURs)"]
    elif route == "Follow-up Required":
        state.triage_reason = "Key case information is missing; reporter follow-up is required before the case can proceed."
        state.next_steps = ["Send AI-generated follow-up questions to the reporter",
                            "Hold the case until minimum information is received",
                            "Re-run triage once the case is complete"]
    elif route == "Quality Review":
        state.triage_reason = "Product quality defect reported; route to Quality Assurance."
        state.next_steps = ["Create a product quality complaint record; capture batch / lot number",
                            "Route to Quality Assurance; check for any co-reported adverse event"]
    else:
        state.triage_reason = "Product information request; route to Medical Information."
        state.next_steps = ["Route to Medical Information; send an approved response",
                            "No Individual Case Safety Report (ICSR) required; log the interaction for traceability"]
    state.case_status = "Awaiting PV Reviewer Decision"
    state.agent_log.append({"agent": "Triage Agent", "status": "completed",
                            "outcome": f"{route} recommended ({priority} priority)"})


def _next_case_id() -> str:
    n = st.session_state.get("case_counter", 0) + 1
    st.session_state.case_counter = n
    return f"DEMO-CASE-{n:04d}"


# ===========================================================================
# STYLING (light enterprise theme)
# ===========================================================================
def inject_css() -> None:
    st.markdown(H("""
    <style>
    :root {--bg:#f4f6fa;--card:#fff;--border:#e4e9f0;--text:#1f2a37;--muted:#6b7a90;
        --blue:#0f6cbd;--teal:#0aa2a2;--green:#16a34a;--amber:#d97706;--red:#dc2626;}
    .stApp {background:var(--bg);color:var(--text);}
    #MainMenu,footer,header {visibility:hidden;}
    .block-container {padding-top:.6rem;padding-bottom:1.1rem;max-width:1320px;}
    .hdr {display:flex;align-items:center;justify-content:space-between;background:var(--card);
        border:1px solid var(--border);border-radius:14px;padding:11px 18px;box-shadow:0 1px 3px rgba(16,24,40,.06);}
    .hdr h1 {font-size:1.3rem;margin:0;font-weight:800;color:#12324f;}
    .hdr .sub {color:var(--muted);font-size:.85rem;margin-top:3px;}
    .badge {display:inline-flex;align-items:center;gap:8px;padding:7px 13px;border-radius:999px;font-size:.8rem;
        font-weight:700;border:1px solid var(--border);background:#f8fafc;}
    .dot {height:9px;width:9px;border-radius:50%;display:inline-block;}
    .dot.green{background:var(--green);} .dot.amber{background:var(--amber);}
    .auth {color:var(--muted);font-size:.72rem;margin-top:6px;text-align:right;}
    .journey {display:flex;align-items:center;justify-content:space-between;gap:4px;background:var(--card);
        border:1px solid var(--border);border-radius:12px;padding:8px 14px;margin:9px 0;}
    .jnode {display:flex;align-items:center;gap:8px;font-size:.8rem;font-weight:700;color:var(--muted);}
    .jnode .c {height:24px;width:24px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;
        font-size:.72rem;background:#eef2f7;color:#8a97a8;border:1px solid var(--border);}
    .jnode.done {color:#12324f;} .jnode.done .c {background:var(--green);color:#fff;border:none;}
    .jnode.active {color:#0f6cbd;} .jnode.active .c {background:var(--blue);color:#fff;border:none;box-shadow:0 0 0 4px rgba(15,108,189,.15);}
    .jarrow {color:#c3cddb;font-weight:800;}
    .card {background:var(--card);border:1px solid var(--border);border-radius:12px;padding:11px 14px;
        box-shadow:0 1px 3px rgba(16,24,40,.05);margin-top:9px;}
    .eyebrow {color:var(--muted);font-size:.7rem;letter-spacing:1.4px;text-transform:uppercase;font-weight:800;margin-bottom:6px;}
    .ch {display:flex;align-items:center;gap:8px;background:#f8fafc;border:1px solid var(--border);border-radius:10px;
        padding:7px 11px;margin:6px 0;font-size:.85rem;} .ch .tag{color:var(--muted);margin-left:auto;font-size:.76rem;}
    .agent {display:flex;gap:10px;align-items:flex-start;padding:6px 0;border-bottom:1px solid #eef2f7;}
    .agent:last-child{border-bottom:none;}
    .agent .ico {height:30px;width:30px;border-radius:8px;display:flex;align-items:center;justify-content:center;
        font-size:1rem;background:#eef4fb;flex-shrink:0;}
    .agent .nm {font-weight:800;color:#1f2a37;font-size:.88rem;}
    .agent .oc {color:var(--muted);font-size:.82rem;margin-top:1px;}
    .tag-status {font-size:.64rem;font-weight:800;letter-spacing:.5px;text-transform:uppercase;padding:2px 7px;
        border-radius:999px;margin-left:6px;}
    .tag-status.completed {background:#e7f6ec;color:#127a32;} .tag-status.attention {background:#fdf1e3;color:#b4650f;}
    .kv {display:flex;justify-content:space-between;gap:10px;padding:3px 0;font-size:.88rem;border-bottom:1px solid #f2f5f9;}
    .kv:last-child{border-bottom:none;} .kv .k {color:var(--muted);} .kv .v {font-weight:700;text-align:right;color:#1f2a37;}
    .chip {display:inline-block;background:#eef4fb;border:1px solid #dbe7f5;color:#0f6cbd;padding:3px 9px;border-radius:999px;font-size:.76rem;margin:2px 4px 2px 0;font-weight:600;}
    .chip.warn {background:#fdf1e3;border-color:#f3dcc0;color:#b4650f;}
    .check {display:flex;align-items:center;justify-content:space-between;padding:7px 0;border-bottom:1px solid #f2f5f9;font-size:.87rem;}
    .check:last-child{border-bottom:none;} .tk{font-weight:800;} .tk.ok{color:var(--green);} .tk.warn{color:var(--amber);} .tk.na{color:#c3cddb;}
    .stat {border:1px solid var(--border);border-radius:10px;padding:10px 12px;background:#fbfcfe;}
    .stat .l{color:var(--muted);font-size:.66rem;letter-spacing:.8px;text-transform:uppercase;font-weight:800;}
    .stat .v{font-size:1.05rem;font-weight:800;margin-top:4px;color:#12324f;}
    .v.serious{color:var(--red);} .v.route{color:var(--blue);}
    .grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;}
    .rec {background:#f0f7ff;border:1px solid #d5e6fb;border-radius:10px;padding:12px 14px;}
    .rec .r {font-size:1.1rem;font-weight:800;color:#0f6cbd;}
    .note {background:#fbfcfe;border:1px dashed var(--border);border-radius:10px;padding:10px 12px;color:#42506e;font-size:.86rem;}
    .steps {margin:4px 0 0;padding-left:18px;} .steps li {font-size:.86rem;margin:3px 0;color:#2a3a52;}
    .outcome {display:flex;align-items:center;gap:9px;padding:6px 0;font-size:.9rem;color:#1f2a37;font-weight:600;}
    .outcome .ok {color:var(--green);font-weight:900;}
    .htable {width:100%;border-collapse:collapse;font-size:.82rem;} .htable th{text-align:left;color:#6b7a90;font-size:.68rem;
        letter-spacing:.6px;text-transform:uppercase;border-bottom:1px solid var(--border);padding:6px 8px;}
    .htable td{padding:6px 8px;border-bottom:1px solid #f2f5f9;} .htable tr:last-child td{border-bottom:none;}
    .sev-Serious{color:var(--red);font-weight:700;} .sev-Non-Serious{color:#127a32;font-weight:600;}
    .exec {background:linear-gradient(90deg,#0f6cbd,#0aa2a2);color:#fff;border-radius:12px;padding:13px 18px;margin-top:14px;
        font-weight:700;text-align:center;font-size:.94rem;}
    .disc {color:var(--muted);font-size:.76rem;text-align:center;margin-top:12px;}
    .metric {background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px;text-align:center;box-shadow:0 1px 3px rgba(16,24,40,.05);}
    .metric .n {font-size:1.7rem;font-weight:800;color:#0f6cbd;} .metric .l {color:var(--muted);font-size:.72rem;letter-spacing:.6px;text-transform:uppercase;margin-top:3px;}
    .brow {display:flex;align-items:center;gap:10px;padding:5px 0;}
    .blabel {width:160px;font-size:.82rem;color:#42506e;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
    .btrack {flex:1;height:14px;background:#eef2f7;border-radius:999px;overflow:hidden;}
    .bfill {height:100%;border-radius:999px;}
    .bval {width:28px;text-align:right;font-weight:800;font-size:.82rem;color:#12324f;}
    .placeholder {border:1px dashed var(--border);border-radius:12px;padding:30px 18px;text-align:center;color:var(--muted);background:#fbfcfe;margin-top:12px;}
    .astrip {display:flex;gap:10px;margin:10px auto 0;max-width:520px;justify-content:center;}
    .achip {flex:1;background:#fff;border:1px solid var(--border);border-radius:12px;padding:12px 8px;text-align:center;box-shadow:0 1px 3px rgba(16,24,40,.05);transition:all .2s;}
    .achip .ai{font-size:1.25rem;} .achip .an{font-weight:800;font-size:.78rem;margin-top:5px;color:#12324f;}
    .achip .as{font-size:.68rem;margin-top:3px;color:var(--muted);min-height:14px;}
    .achip.pending{opacity:.5;}
    .achip.working{border-color:var(--blue);box-shadow:0 0 0 3px rgba(15,108,189,.12);}
    .achip.working .as{color:var(--blue);font-weight:700;}
    .achip.done{border-color:#bfe6cb;} .achip.done .as{color:#127a32;font-weight:700;}
    .achip.attention{border-color:#f3dcc0;} .achip.attention .as{color:#b4650f;font-weight:700;}
    .closebanner{background:#e7f6ec;border:1px solid #bfe6cb;border-radius:12px;padding:14px 16px;margin-top:12px;}
    .closebanner .t{font-weight:800;color:#127a32;font-size:1.02rem;}
    .dispo{list-style:none;padding:0;margin:6px 0 0;} .dispo li{padding:6px 0 6px 24px;position:relative;font-size:.86rem;color:#2a3a52;border-bottom:1px solid #f2f5f9;}
    .dispo li:last-child{border-bottom:none;} .dispo li:before{content:"\\2192";position:absolute;left:2px;color:#0f6cbd;font-weight:800;}
    .dispo li.done:before{content:"\\2713";color:#16a34a;}
    .stTextArea textarea {background:#fff !important;color:#1f2a37 !important;border:1px solid var(--border) !important;border-radius:10px !important;}
    .stTextArea label,.stSelectbox label {color:#42506e !important;font-weight:700 !important;}
    div[data-baseweb="select"] > div {background:#fff !important;border:1px solid var(--border) !important;border-radius:10px !important;}
    div.stButton>button {border-radius:10px;font-weight:700;border:1px solid var(--border);background:#fff;color:#1f2a37;}
    div.stButton>button:hover {border-color:var(--blue);color:var(--blue);}
    div.stButton>button[kind="primary"] {background:var(--blue);border:none;color:#fff;box-shadow:0 2px 6px rgba(15,108,189,.3);}
    .stTabs [data-baseweb="tab-list"] {gap:4px;} .stTabs [data-baseweb="tab"] {font-weight:700;font-size:.85rem;}
    </style>"""), unsafe_allow_html=True)


# ===========================================================================
# RENDER — shared
# ===========================================================================
def render_header(connected: bool) -> None:
    badge = ('<span class="badge"><span class="dot green"></span> Azure AI Connected</span>' if connected
             else '<span class="badge"><span class="dot amber"></span> Simulation Mode</span>')
    st.markdown(H(f"""
        <div class="hdr">
          <div><h1>Agentic Pharmacovigilance</h1></div>
          <div style="text-align:right;">{badge}<div class="auth">Authentication: Microsoft Entra ID</div></div>
        </div>"""), unsafe_allow_html=True)


def render_journey(done: int, active: int) -> None:
    row = '<div class="journey">'
    for i, name in enumerate(JOURNEY, start=1):
        cls = "done" if i <= done else ("active" if i == active else "")
        mark = "\u2713" if i <= done else i
        row += f'<div class="jnode {cls}"><span class="c">{mark}</span>{esc(name)}</div>'
        if i < len(JOURNEY):
            row += '<span class="jarrow">\u203a</span>'
    st.markdown(H(row + "</div>"), unsafe_allow_html=True)


AGENT_ICON = {"Intake Agent": "\U0001F4E5", "Extraction Agent": "\U0001F9EC",
              "History Agent": "\U0001F5C3\uFE0F", "Validation Agent": "\U0001F50E", "Triage Agent": "\U0001F9ED"}


def render_agent_pipeline(state: PVCaseState) -> None:
    rows = ""
    for a in state.agent_log:
        ico = AGENT_ICON.get(a["agent"], "\U0001F916")
        lbl = "Completed" if a["status"] == "completed" else "Attention"
        rows += (f'<div class="agent"><div class="ico">{ico}</div><div style="flex:1;">'
                 f'<div class="nm">{esc(a["agent"])}<span class="tag-status {a["status"]}">{lbl}</span></div>'
                 f'<div class="oc">{esc(a["outcome"])}</div></div></div>')
    st.markdown(H(f'<div class="card"><div class="eyebrow">AI Agent Team — sequential handoff</div>{rows}</div>'),
                unsafe_allow_html=True)


# ===========================================================================
# RENDER — result tabs
# ===========================================================================
def tab_case_triage(state: PVCaseState) -> None:
    ev = "".join(f'<span class="chip">{esc(e)}</span>' for e in state.adverse_events) or '<span class="v">None reported</span>'
    p, prod = state.patient, state.product
    prod_txt = (esc(prod.get("name")) + ((" " + esc(prod.get("strength"))) if prod.get("strength") else "")) if prod.get("name") else "Not reported"
    st.markdown(H(f"""
        <div class="card"><div class="eyebrow">Case Summary — {esc(state.case_id)}</div>
          <div class="kv"><span class="k">Product</span><span class="v">{prod_txt}</span></div>
          <div class="kv"><span class="k">Adverse event(s)</span><span class="v">{ev}</span></div>
          <div class="kv"><span class="k">Seriousness</span><span class="v">{esc(state.seriousness)}</span></div>
          <div class="kv"><span class="k">Reporter / Patient</span><span class="v">{esc(state.reporter.get('reporter_type'))} · {esc(p.get('age'))}/{esc(p.get('gender'))}</span></div>
          <div class="kv"><span class="k">Onset / Hospitalized</span><span class="v">{esc(state.onset)} · {esc(state.hospitalization)}</span></div>
        </div>"""), unsafe_allow_html=True)
    ser_cls = "serious" if state.seriousness == "Serious" else ""
    st.markdown(H(f"""
        <div class="card"><div class="eyebrow">Triage Agent Recommendation</div>
          <div class="grid3">
            <div class="stat"><div class="l">Classification</div><div class="v">{esc(state.category)}</div></div>
            <div class="stat"><div class="l">Seriousness</div><div class="v {ser_cls}">{esc(state.seriousness)}</div></div>
            <div class="stat"><div class="l">Priority</div><div class="v">{esc(state.triage_priority)}</div></div>
            <div class="stat"><div class="l">Route</div><div class="v route">{esc(state.recommended_route)}</div></div>
            <div class="stat"><div class="l">Confidence</div><div class="v">{state.confidence}%</div></div>
            <div class="stat"><div class="l">Initial Signal</div><div class="v">{esc(state.signal_type)}</div></div>
          </div>
          <div class="note" style="margin-top:10px;">{esc(state.triage_reason)} <span style="color:#6b7a90;">(downstream actions available in Human Review &amp; Closure)</span></div>
        </div>"""), unsafe_allow_html=True)


def tab_validation_history(state: PVCaseState) -> None:
    sym = {"ok": ("\u2713", "ok"), "warn": ("\u26a0", "warn"), "na": ("\u2014", "na")}
    rows = "".join(f'<div class="check"><span>{esc(c["label"])}</span>'
                   f'<span class="tk {sym[c["status"]][1]}">{sym[c["status"]][0]}</span></div>'
                   for c in state.validation_checks)
    if state.missing_information:
        action = (f'<div class="note" style="margin-top:8px;"><strong>Follow-up recommended</strong> — the case is missing '
                  f'{", ".join(esc(m) for m in state.missing_information)}. It cannot be finalised until the reporter '
                  f'provides this, so the Triage Agent routes it as <strong>Follow-up Required</strong> and the PV '
                  f'reviewer can send the questions.</div>')
    else:
        action = '<div class="note" style="margin-top:8px;">All minimum information present — the case can proceed to routing.</div>'
    st.markdown(H(f'<div class="card"><div class="eyebrow">Validation Agent — completeness</div>{rows}{action}</div>'),
                unsafe_allow_html=True)

    if state.similar_cases:
        trs = "".join(f'<tr><td>{esc(h["case_id"])}</td><td>{esc(h["date"])}</td><td>{esc(h["event"])}</td>'
                      f'<td class="sev-{esc(h["seriousness"])}">{esc(h["seriousness"])}</td>'
                      f'<td>{esc(h["route"])}</td><td>{esc(h["outcome"])}</td></tr>' for h in state.similar_cases)
        serious_prior = sum(1 for h in state.similar_cases if h["seriousness"] == "Serious")
        banner = (f'<div class="note" style="margin-bottom:8px;">Found <strong>{len(state.similar_cases)}</strong> similar '
                  f'prior case(s) for <strong>{esc(state.product.get("name"))}</strong>'
                  + (f' — <strong style="color:#b4650f;">{serious_prior} previously serious</strong>, a potential safety signal to monitor.'
                     if serious_prior else '.') + '</div>')
        st.markdown(H(f'<div class="card"><div class="eyebrow">History Agent — similar prior cases</div>{banner}'
                      f'<table class="htable"><tr><th>Case</th><th>Date</th><th>Event</th><th>Seriousness</th>'
                      f'<th>Route</th><th>Outcome</th></tr>{trs}</table></div>'), unsafe_allow_html=True)
    else:
        st.markdown(H('<div class="card"><div class="eyebrow">History Agent — similar prior cases</div>'
                      '<div class="note">No similar prior cases were found for this product / category.</div></div>'),
                    unsafe_allow_html=True)


def _start_new_case_button(key: str) -> None:
    st.markdown('<div style="text-align:center;color:#127a32;font-weight:800;margin-top:10px;">'
                '\u2713 Workflow complete \u2014 all agent steps and human review finished</div>',
                unsafe_allow_html=True)
    if st.button("\U0001F3E0 Start New Case", type="primary", use_container_width=True, key=key):
        for k in ("state", "show_change", "fallback_reason", "mode"):
            st.session_state.pop(k, None)
        st.session_state["_reload_msg"] = True
        st.rerun()


def tab_review(state: PVCaseState) -> None:
    if state.human_decision is None:
        st.markdown(H(f"""
            <div class="card"><div class="eyebrow">Human PV Review — AI recommends, human decides</div>
              <div class="rec"><div style="color:#6b7a90;font-size:.72rem;letter-spacing:1px;text-transform:uppercase;font-weight:800;">AI Recommendation</div>
                <div class="r">{esc(state.recommended_route)}</div>
                <div style="color:#42506e;margin-top:4px;">{esc(state.triage_reason)} · Confidence {state.confidence}%</div>
              </div></div>"""), unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        if c1.button("\u2705 Accept", type="primary", use_container_width=True):
            state.human_decision = f"Accepted — {state.recommended_route}"
            state.case_status = "Human Review Completed — Ready for Next Step"
            st.session_state.state = state
            st.rerun()
        if c2.button("\u2753 Request Info", use_container_width=True):
            state.human_decision = "Information Requested"
            state.case_status = "On Hold — Awaiting Reporter Follow-up"
            st.session_state.state = state
            st.rerun()
        if c3.button("\U0001F503 Change Route", use_container_width=True):
            st.session_state.show_change = True
        st.caption("**Accept** = approve the AI route · **Request Info** = send follow-up questions to the reporter "
                   "(the human action that resolves a *Follow-up Required* case) · **Change Route** = override the AI.")
        if st.session_state.get("show_change"):
            opts = ["Safety Review", "Quality Review", "Medical Information", "Follow-up Required"]
            new = st.selectbox("Correct route", opts,
                               index=opts.index(state.recommended_route) if state.recommended_route in opts else 0)
            if st.button("Confirm change", type="primary"):
                state.recommended_route = new
                state.human_decision = f"Route changed — {new}"
                state.case_status = "Human Review Completed — Ready for Next Step"
                st.session_state.show_change = False
                st.session_state.state = state
                st.rerun()
        return

    # Outcome
    if state.human_decision == "Information Requested":
        qmap = {"Event onset date": "Please provide the date when the symptoms first occurred.",
                "Patient outcome": "Please provide the current outcome of the reported event.",
                "Patient details": "Please provide the patient's age and gender.",
                "Suspect product": "Which product, brand and strength was taken?",
                "Adverse event": "What specific symptoms or reactions were experienced?"}
        qs = [qmap[m] for m in state.missing_information if m in qmap] or [
            "Please provide any additional details about the report."]
        default_text = "\n".join(qs)
        fu_key = f"fu_{state.case_id}"
        if not state.closed:
            st.session_state.setdefault(fu_key, default_text)
            st.markdown(H('<div class="card"><div class="eyebrow">Follow-up to the Reporter</div>'
                          '<div class="note">Edit the AI-generated questions or write your own before sending — '
                          'one question per line.</div></div>'), unsafe_allow_html=True)
            st.text_area("Follow-up questions", key=fu_key, height=130)
            b1, b2 = st.columns([1, 1])
            b1.button("\u21ba Use AI-generated", use_container_width=True,
                      on_click=lambda: st.session_state.__setitem__(fu_key, default_text))
            if b2.button("\U0001F4E4 Send Follow-up & Hold Case", type="primary", use_container_width=True):
                state.followup_text = st.session_state.get(fu_key, default_text).strip()
                state.closed = True
                state.case_status = "On Hold — Follow-up sent, awaiting reporter response"
                state.disposition = "Follow-up request issued to the reporter; case held pending response."
                st.session_state.state = state
                st.rerun()
        else:
            sent = [q for q in (state.followup_text or default_text).split("\n") if q.strip()]
            body = "".join(f'<div class="outcome"><span class="ok">?</span>{esc(q)}</div>' for q in sent)
            st.markdown(H(f'<div class="card"><div class="eyebrow">Outcome — Follow-up Sent</div>'
                          f'<div class="note">The following questions were sent to the reporter:</div>{body}'
                          f'<div style="margin-top:8px;color:#6b7a90;">Case status: {esc(state.case_status)}</div></div>'),
                        unsafe_allow_html=True)
            st.markdown(H(f'<div class="closebanner"><div class="t">\u2713 Case Held — Follow-up Sent</div>'
                          f'<div style="margin-top:4px;color:#2a3a52;">{esc(state.disposition)}</div></div>'),
                        unsafe_allow_html=True)
            _start_new_case_button("new_case_hold")
        return

    # Accepted / route-changed path
    st.markdown(H(f"""
        <div class="card"><div class="eyebrow">Reviewer Decision Recorded</div>
          <div class="outcome"><span class="ok">\u2713</span> Signal understood (Intake Agent)</div>
          <div class="outcome"><span class="ok">\u2713</span> Case structured (Extraction Agent)</div>
          <div class="outcome"><span class="ok">\u2713</span> Prior cases checked (History Agent)</div>
          <div class="outcome"><span class="ok">\u2713</span> Case validated (Validation Agent)</div>
          <div class="outcome"><span class="ok">\u2713</span> Triage completed — {esc(state.recommended_route)}</div>
          <div class="outcome"><span class="ok">\u2713</span> Reviewer decision — {esc(state.human_decision)}</div>
        </div>"""), unsafe_allow_html=True)

    if not state.closed:
        done_items = "".join(f"<li>{esc(s)}</li>" for s in state.next_steps)
        st.markdown(H(f'<div class="card"><div class="eyebrow">Case Disposition — downstream actions</div>'
                      f'<div class="note">Execute the recommended actions to complete and close the case.</div>'
                      f'<ul class="dispo">{done_items}</ul></div>'), unsafe_allow_html=True)
        if st.button("\u2699\uFE0F Execute Downstream Actions & Close Case", type="primary"):
            _close_case(state)
            st.session_state.state = state
            st.rerun()
    else:
        done_items = "".join(f'<li class="done">{esc(s)}</li>' for s in state.next_steps)
        st.markdown(H(f'<div class="card"><div class="eyebrow">Case Disposition — completed</div>'
                      f'<ul class="dispo">{done_items}</ul></div>'), unsafe_allow_html=True)
        st.markdown(H(f'<div class="closebanner"><div class="t">\u2713 Case Closed — {esc(state.case_status)}</div>'
                      f'<div style="margin-top:4px;color:#2a3a52;">{esc(state.disposition)}</div></div>'),
                    unsafe_allow_html=True)
        _start_new_case_button("new_case_done")


def _close_case(state: PVCaseState) -> None:
    """Simulate execution of downstream actions and close the case."""
    icsr = state.case_id.replace("DEMO-CASE", "DEMO-ICSR")
    route = state.recommended_route
    if route == "Safety Review":
        clock = ("expedited 15-day reporting clock started" if state.seriousness == "Serious"
                 else "included in Periodic Benefit-Risk Evaluation Reports (PBRERs) / "
                      "Periodic Safety Update Reports (PSURs)")
        state.disposition = (f"Individual Case Safety Report (ICSR) {icsr} created · E2B(R3) electronic draft "
                             f"generated · routed to Safety Review ({state.triage_priority} priority) · {clock}.")
        state.case_status = "Closed — Submitted to Safety Review"
    elif route == "Quality Review":
        state.disposition = (f"Product quality complaint {icsr} created · batch/lot flagged · "
                             "routed to Quality Assurance.")
        state.case_status = "Closed — Routed to Quality"
    elif route == "Medical Information":
        state.disposition = "Approved medical information response sent · interaction logged · no Individual Case Safety Report (ICSR) required."
        state.case_status = "Closed — Medical Information Responded"
    else:  # Follow-up Required accepted
        state.disposition = "Follow-up workflow initiated; case will re-enter triage once information is received."
        state.case_status = "Closed — Follow-up Initiated"
    state.actions_done = list(state.next_steps)
    state.closed = True


# ===========================================================================
# PAGE 1 — COMMAND CENTER
# ===========================================================================
def _load_message() -> None:
    """Load the message for the current Scenario x Channel combination and reset the case."""
    scn = st.session_state.get("scn_sel")
    ch = st.session_state.get("chan_sel")
    msg = MESSAGES.get((scn, ch))
    if msg:
        st.session_state.message_area = msg
    st.session_state.state = None
    st.session_state.pop("show_change", None)


def _on_scenario_change() -> None:
    _load_message()


def _on_channel_change() -> None:
    _load_message()


def page_command_center(azure, connected: bool) -> None:
    # Reload the default message for the current scenario/channel on a fresh case
    # (done here, before the text box is created, so setting its value is safe).
    if st.session_state.pop("_reload_msg", False):
        msg = MESSAGES.get((st.session_state.get("scn_sel"), st.session_state.get("chan_sel")))
        if msg:
            st.session_state["message_area"] = msg
    state: Optional[PVCaseState] = st.session_state.state
    if state is None:
        render_journey(0, 1)
    elif state.human_decision is None:
        render_journey(4, 5)
    elif not state.closed:
        render_journey(5, 6)
    else:
        render_journey(6, 6)

    left, right = st.columns([5, 7], gap="large")

    with left:
        st.markdown('<div class="eyebrow">Safety Signal Intake</div>', unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        with sc1:
            st.selectbox("Scenario", list(SCENARIOS.keys()), key="scn_sel", on_change=_on_scenario_change)
        with sc2:
            channel = st.selectbox("Channel", CHANNELS, key="chan_sel", on_change=_on_channel_change)
        st.markdown(H(f'<div class="ch"><span>{CHANNEL_ICON.get(channel, "")}</span> <strong>{esc(channel)}</strong></div>'),
                    unsafe_allow_html=True)
        message = st.text_area("Incoming message", height=120, key="message_area")
        b1, b2 = st.columns([3, 2])
        if b1.button("\u26a1 Process Safety Signal", type="primary", use_container_width=True):
            if message.strip():
                st.session_state._go = True
                st.rerun()
            else:
                st.warning("Enter a message before processing.")
        if state is not None and b2.button("\U0001F504 New Case", use_container_width=True):
            for k in ("state", "show_change"):
                st.session_state.pop(k, None)
            st.session_state["_reload_msg"] = True
            st.rerun()

        if st.session_state.state is not None:
            render_agent_pipeline(st.session_state.state)

    with right:
        state = st.session_state.state
        if st.session_state.pop("_go", False):
            msg = st.session_state.message_area
            ch = st.session_state.chan_sel
            if msg.strip():
                _run_agents(msg, ch, azure, connected)  # renders the agent strip here + reruns
            else:
                st.warning("Enter a message before processing.")
        elif state is None:
            st.empty()
        else:
            mode = st.session_state.get("mode", "sim")
            if mode != "azure":
                st.caption("\U0001F9EA Simulation Mode (deterministic local responses).")
                if st.session_state.get("fallback_reason"):
                    with st.expander("Why Simulation Mode?"):
                        st.code(st.session_state.fallback_reason)
            t1, t2, t3 = st.tabs(["Case & Triage", "Validation & History", "Human Review & Closure"])
            with t1:
                tab_case_triage(state)
            with t2:
                tab_validation_history(state)
            with t3:
                tab_review(state)
            with st.expander("Shared Case State (JSON)"):
                st.json(json.loads(state.model_dump_json()))


def _run_agents(message, channel, azure, connected) -> None:
    """Run the agent team as a centered progress strip that updates in place."""
    agents = [("Intake Agent", "\U0001F4E5"), ("Extraction Agent", "\U0001F9EC"),
              ("History Agent", "\U0001F5C3\uFE0F"), ("Validation Agent", "\U0001F50E"),
              ("Triage Agent", "\U0001F9ED")]
    statuses = ["pending"] * 5
    notes = ["Waiting"] * 5
    ph = st.empty()

    def _draw():
        chips = "".join(
            f'<div class="achip {statuses[i]}"><div class="ai">{agents[i][1]}</div>'
            f'<div class="an">{esc(agents[i][0])}</div><div class="as">{esc(notes[i])}</div></div>'
            for i in range(5))
        ph.markdown(H(f'<div class="astrip">{chips}</div>'), unsafe_allow_html=True)

    def chip(i, status, note):
        statuses[i] = status
        notes[i] = note
        _draw()

    for i in range(5):
        chip(i, "pending", "Waiting")
    prog = st.progress(0)
    state = PVCaseState(case_id=_next_case_id(), channel=channel, raw_message=message)

    chip(0, "working", "Analyzing signal\u2026")
    time.sleep(0.5)
    intake, m1, r1 = intake_agent(message, channel, azure, connected)
    apply_intake(state, intake)
    chip(0, "done", state.agent_log[-1]["outcome"])
    prog.progress(20)

    chip(1, "working", "Structuring case\u2026")
    time.sleep(0.4)
    extr, m2, r2 = extraction_agent(message, channel, intake, azure, connected)
    apply_extraction(state, extr)
    chip(1, "done", state.agent_log[-1]["outcome"])
    prog.progress(40)

    chip(2, "working", "Searching history\u2026")
    time.sleep(0.7)
    history_agent(state)
    chip(2, "attention" if state.agent_log[-1]["status"] == "attention" else "done",
         state.agent_log[-1]["outcome"])
    prog.progress(60)

    chip(3, "working", "Checking gaps\u2026")
    time.sleep(0.7)
    validation_agent(state)
    chip(3, "attention" if state.agent_log[-1]["status"] == "attention" else "done",
         state.agent_log[-1]["outcome"])
    prog.progress(80)

    chip(4, "working", "Deciding route\u2026")
    time.sleep(0.7)
    triage_agent(state)
    chip(4, "done", state.agent_log[-1]["outcome"])
    prog.progress(100)
    time.sleep(0.5)

    mode = "azure" if (m1 == "azure" and m2 == "azure") else "sim"
    st.session_state.state = state
    st.session_state.mode = mode
    st.session_state.fallback_reason = r1 or r2
    st.session_state.show_change = False
    st.session_state.session_cases.append({
        "case_id": state.case_id, "date": datetime.now().strftime("%d %b %Y"),
        "product": state.product.get("name") or "Unknown",
        "event": (state.adverse_events[0] if state.adverse_events else state.category),
        "category": state.category, "seriousness": state.seriousness or "Unknown",
        "route": state.recommended_route, "outcome": state.outcome, "source": "This session"})
    st.rerun()


# ===========================================================================
# PAGE 2 — ENTERPRISE DASHBOARD
# ===========================================================================
def page_dashboard() -> None:
    from collections import Counter

    cases = st.session_state.session_cases + HISTORY
    total = len(cases)
    serious = sum(1 for c in cases if c["seriousness"] == "Serious")
    safety = sum(1 for c in cases if c["route"] == "Safety Review")
    quality = sum(1 for c in cases if c["route"] == "Quality Review")
    medinfo = sum(1 for c in cases if c["route"] == "Medical Information")
    this_session = len(st.session_state.session_cases)

    st.markdown('<div class="eyebrow" style="margin-top:6px;">Enterprise PV Portfolio (synthetic)</div>', unsafe_allow_html=True)
    hc1, hc2 = st.columns([4, 1])
    hc1.caption("Historical seed data plus cases you process this session.")
    if hc2.button("\U0001F9F9 Clear Session Numbers", use_container_width=True):
        st.session_state.session_cases = []
        st.session_state.case_counter = 0
        st.rerun()
    cols = st.columns(6)
    for col, (label, n) in zip(cols, [("Total Cases", total), ("Serious", serious), ("Safety Review", safety),
                                      ("Quality Review", quality), ("Medical Info", medinfo),
                                      ("This Session", this_session)]):
        col.markdown(H(f'<div class="metric"><div class="n">{n}</div><div class="l">{label}</div></div>'),
                     unsafe_allow_html=True)

    def bars(title, counts, color):
        if not counts:
            return f'<div class="card"><div class="eyebrow">{esc(title)}</div><div class="note">No data.</div></div>'
        mx = max(counts.values())
        rows = ""
        for label, n in sorted(counts.items(), key=lambda x: -x[1]):
            w = int(round(100 * n / mx)) if mx else 0
            rows += (f'<div class="brow"><div class="blabel">{esc(label)}</div>'
                     f'<div class="btrack"><div class="bfill" style="width:{w}%;background:{color};"></div></div>'
                     f'<div class="bval">{n}</div></div>')
        return f'<div class="card"><div class="eyebrow">{esc(title)}</div>{rows}</div>'

    route_c = Counter(c["route"] for c in cases)
    prod_c = Counter(c["product"] for c in cases)
    ser_c = Counter(c["seriousness"] for c in cases)
    cat_c = Counter(c["category"] for c in cases)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(H(bars("Cases by Route", route_c, "#0f6cbd")), unsafe_allow_html=True)
        st.markdown(H(bars("Cases by Product", prod_c, "#0aa2a2")), unsafe_allow_html=True)
    with right:
        st.markdown(H(bars("Cases by Seriousness", ser_c, "#d97706")), unsafe_allow_html=True)
        st.markdown(H(bars("Cases by Category", cat_c, "#16a34a")), unsafe_allow_html=True)

    trs = "".join(
        f'<tr><td>{esc(c["case_id"])}</td><td>{esc(c["date"])}</td><td>{esc(c["product"])}</td>'
        f'<td>{esc(c["event"])}</td><td>{esc(c["category"])}</td>'
        f'<td class="sev-{esc(c["seriousness"])}">{esc(c["seriousness"])}</td>'
        f'<td>{esc(c["route"])}</td><td>{esc(c["source"])}</td></tr>'
        for c in cases[:15])
    st.markdown(H('<div class="card"><div class="eyebrow">Recent Cases</div>'
                  '<table class="htable"><tr><th>Case ID</th><th>Date</th><th>Product</th><th>Event</th>'
                  '<th>Category</th><th>Seriousness</th><th>Route</th><th>Source</th></tr>'
                  + trs + '</table></div>'), unsafe_allow_html=True)
    st.caption("Illustrative synthetic portfolio — historical seed data plus cases processed this session.")


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    st.set_page_config(page_title="Agentic Pharmacovigilance AI", page_icon="\U0001F9EC", layout="wide")
    inject_css()
    st.session_state.setdefault("state", None)
    st.session_state.setdefault("chan_sel", "Patient WhatsApp")
    st.session_state.setdefault("scn_sel", "Serious Adverse Event")
    st.session_state.setdefault("message_area", MESSAGES[("Serious Adverse Event", "Patient WhatsApp")])
    st.session_state.setdefault("session_cases", [])
    st.session_state.setdefault("page", "Case Command Center")

    azure = get_azure_client()
    connected = azure is not None

    render_header(connected)
    page = st.radio("nav", ["Case Command Center", "Enterprise Dashboard"], horizontal=True,
                    label_visibility="collapsed", key="page")

    if page == "Case Command Center":
        page_command_center(azure, connected)
    else:
        page_dashboard()

    st.markdown(H('<div class="disc">Demonstration only — all data and product names are dummy and fictional. '
                  'Outputs are illustrative and require qualified '
                  'pharmacovigilance review before operational or regulatory use.</div>'), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
