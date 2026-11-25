"""LLM layer: (1) clinical information extraction from free text, (2) clinician-style report.

Provider: Groq (OpenAI-compatible) by default - set GROQ_API_KEY. Any other OpenAI-compatible
endpoint can be used with LLM_PROVIDER=openai, OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL.
Privacy: patient name / screening ID are never sent to the LLM.
If the LLM is unreachable, the app falls back to the rule-based extractor and a template report.
"""
from __future__ import annotations

import json
import os

from .config import LLM_MODEL, LLM_PROVIDER

_client = None

CANONICAL_CONDITIONS = ["diabetes", "hypertension", "cardiovascular_disease", "stroke", "thyroid",
                        "cancer", "kidney_disease", "dyslipidemia"]
CANONICAL_SYMPTOMS = ["polyuria", "polydipsia", "polyphagia", "weight_loss", "fatigue", "blurred_vision",
                      "numbness_tingling", "slow_healing", "headache", "chest_pain"]
MED_CLASSES = ["diabetes_medication", "antihypertensive", "lipid_lowering", "antiplatelet", "thyroid_medication", "other"]


def available() -> bool:
    if LLM_PROVIDER == "groq":
        return bool(os.getenv("GROQ_API_KEY"))
    return bool(os.getenv("OPENAI_API_KEY"))


def _get_client():
    global _client
    if _client is None:
        if LLM_PROVIDER == "groq":
            from groq import Groq
            _client = Groq(timeout=60)
        else:
            from openai import OpenAI
            _client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None, timeout=60)
    return _client


def _chat(system: str, user: str, json_mode: bool = False, max_tokens: int = 2000) -> str:
    kw = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = _get_client().chat.completions.create(
        model=LLM_MODEL, temperature=0.1, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
    return resp.choices[0].message.content or ""


# ------------------------------------------------------------------ extraction
EXTRACT_SYSTEM = f"""You are a clinical NLP engine. Extract information from a clinician's free-text
screening notes. Handle negation ("no", "denies", "nil") and abbreviations (DM, HTN, CVD, CA, IHD, CVA,
OD/BD/TDS). Only report what is written - never infer a diagnosis that is not stated.
Return ONLY a JSON object with exactly these keys:
{{
 "conditions": [canonical condition keys present in the PATIENT's own history/complaints],
 "negated_conditions": [canonical keys explicitly denied for the patient],
 "symptoms": [canonical symptom keys present],
 "negated_symptoms": [canonical symptom keys explicitly denied],
 "medications": [{{"name": str, "dose": str|null, "frequency": str|null, "class": one of {MED_CLASSES}}}],
 "family_history": [{{"relation": str, "condition": canonical condition key}}],
 "family_negated": [canonical keys explicitly denied in the family],
 "labs": {{"hba1c_percent": number|null, "fasting_glucose_mgdl": number|null, "postprandial_glucose_mgdl": number|null}},
 "sex": "male"|"female"|null,
 "other_findings": [short strings for anything else clinically relevant, e.g. vision or lower-limb findings]
}}
Canonical condition keys: {CANONICAL_CONDITIONS}
Canonical symptom keys: {CANONICAL_SYMPTOMS}
Convert glucose given in mmol/L to mg/dL (x18)."""


def extract(sections: dict) -> dict:
    user = "\n".join(f"[{k.replace('_', ' ').title()}]\n{v or '(empty)'}" for k, v in sections.items())
    data = json.loads(_chat(EXTRACT_SYSTEM, user, json_mode=True))

    def lst(k):
        v = data.get(k) or []
        return [x for x in v if isinstance(x, (str, dict))]

    conds = set(c for c in lst("conditions") if isinstance(c, str))
    meds = [m for m in lst("medications") if isinstance(m, dict)]
    med_classes = sorted({m.get("class") for m in meds if m.get("class")})
    fam = [f for f in lst("family_history") if isinstance(f, dict)]
    fam_conds = {f.get("condition") for f in fam}
    syms = [s for s in lst("symptoms") if isinstance(s, str)]

    def provided(*keys):
        return any((sections.get(k) or "").strip() for k in keys)

    hist_given = provided("complaints", "existing_illness", "medications")
    fam_given = provided("family_history")

    def flag(present, given):
        return 1.0 if present else (0.0 if given else None)

    features = {
        "known_diabetes": flag("diabetes" in conds or "diabetes_medication" in med_classes, hist_given),
        "on_diabetes_medication": flag("diabetes_medication" in med_classes, provided("medications")),
        "hypertensive": flag("hypertension" in conds or "antihypertensive" in med_classes, hist_given),
        "cardiovascular_disease": flag("cardiovascular_disease" in conds, hist_given),
        "stroke": flag("stroke" in conds, hist_given),
        "family_diabetes": flag("diabetes" in fam_conds, fam_given),
        "family_hypertension": flag("hypertension" in fam_conds, fam_given),
        "symptom_count": float(len(syms)),
        "classic_hyperglycaemia_symptoms": float(len({"polyuria", "polydipsia", "polyphagia", "weight_loss"} & set(syms))),
    }
    labs = {k: v for k, v in (data.get("labs") or {}).items() if isinstance(v, (int, float))}
    return {
        "method": f"LLM ({LLM_MODEL})",
        "conditions": sorted(conds), "negated_conditions": sorted(set(lst("negated_conditions")) - conds),
        "condition_evidence": {},
        "symptoms": sorted(set(syms)), "negated_symptoms": sorted(set(lst("negated_symptoms"))),
        "medication_classes": [c for c in med_classes if c != "other"],
        "medications": [{"text": m.get("name"), "classes": [m.get("class")] if m.get("class") else [],
                         "dose": m.get("dose"), "frequency": m.get("frequency")} for m in meds],
        "family_history": [{"condition": f.get("condition"), "relation": f.get("relation") or "unspecified"} for f in fam],
        "family_negated": sorted(set(lst("family_negated"))),
        "labs": labs, "sex": data.get("sex") if data.get("sex") in ("male", "female") else None,
        "other_findings": [x for x in lst("other_findings") if isinstance(x, str)],
        "features": features,
    }


# ------------------------------------------------------------------ report
REPORT_SYSTEM = """You are an experienced consultant physician (internal medicine / diabetology) writing a
clinical assessment note for a fellow clinician, based on a screening form and the output of validated
machine-learning models. Write in plain text (no markdown symbols like ** or #), in the style of a doctor's
case note, using clear section headings in CAPITALS followed by a colon. Be precise and concise.

Sections, in this order:
CLINICAL SUMMARY: one short paragraph (age, key findings, measurements, BMI).
DIABETES ASSESSMENT: state the status given in the input (known diabetic / meets a diagnostic criterion /
  model-estimated risk / insufficient data). Quote the model score and how many models agree. Interpret
  GRBS against ADA criteria (random >=200 mg/dL with classic symptoms is diagnostic; otherwise needs FPG
  >=126 mg/dL or HbA1c >=6.5% or 2-h OGTT >=200 mg/dL, confirmed). Never state a definitive new diagnosis
  from a model score alone.
HYPERTENSION ASSESSMENT: BP category (ACC/AHA), model score, known history.
BMI AND METABOLIC RISK: WHO category and the Asian cut-offs (overweight >=23, obese >=27.5).
KEY CONTRIBUTING FACTORS: from the explainability output (SHAP/LIME), in clinical language.
MEDICATION REVIEW: list each current medication with dose and frequency as recorded and comment on
  relevance. Then explain the rationale for each item in "draft_prescription" (already safety-checked;
  the formatted prescription is appended after your note, so do NOT rewrite it or add other drugs).
  Mention any items removed by the safety check and any warnings.
RECOMMENDED INVESTIGATIONS: e.g. FPG, HbA1c, lipid profile, serum creatinine/eGFR, urine ACR, ECG,
  fundus examination - only what is relevant.
LIFESTYLE ADVICE AND FOLLOW-UP: concise and specific; incorporate the clinician's own advice and
  follow-up text if given.
DATA QUALITY AND LIMITATIONS: missing or implausible inputs, abstentions, model limitations.
End with the line: "AI-assisted decision support - to be verified by the treating clinician."
"""


def generate_report(payload: dict) -> str:
    return _chat(REPORT_SYSTEM, "Screening data and model outputs (JSON):\n" + json.dumps(payload, indent=1, default=str),
                 max_tokens=3000).strip()
