"""Draft prescription: LLM-written, checked against a guideline formulary, with a rule-based fallback.

Safety design
* A drug is prescribed only for a condition that is DOCUMENTED or meets DIAGNOSTIC CRITERIA,
  never for a model risk score alone (risk only -> lifestyle + confirmatory tests).
* Every LLM-proposed item must be in the formulary, inside its dose range and allowed frequency,
  and not blocked by a contraindication. Anything else is removed and the reason is shown.
* Existing medications are continued (the clinician may stop/modify them).
* Output is a DRAFT that the clinician must review and approve before printing.
Sources: ADA Standards of Care 2025 (Sec. 9, 10), ACC/AHA 2017 BP guideline, ESC/ADA lipid guidance.
"""
from __future__ import annotations

import json
import re

FREQ = {  # code -> (morning-afternoon-night pattern, plain text)
    "OD": ("1-0-0", "once daily (morning)"),
    "OD-N": ("0-0-1", "once daily (night)"),
    "HS": ("0-0-1", "at bedtime"),
    "BD": ("1-0-1", "twice daily"),
    "TDS": ("1-1-1", "three times daily"),
}

# drug -> class, strengths (mg), per-dose range, allowed freqs, max/day, timing, indication key
FORMULARY = {
    "Metformin": {"class": "diabetes_medication", "strengths": [500, 850, 1000], "dose": (500, 1000), "freq": ["OD", "BD"],
                  "max_day": 2000, "timing": "after food", "for": "diabetes",
                  "note": "Start low, increase after 1-2 weeks to reduce GI upset. Check eGFR/B12 periodically."},
    "Glimepiride": {"class": "diabetes_medication", "strengths": [1, 2], "dose": (1, 4), "freq": ["OD"], "max_day": 8,
                    "timing": "before breakfast", "for": "diabetes", "note": "Risk of hypoglycaemia - counsel; caution in elderly."},
    "Sitagliptin": {"class": "diabetes_medication", "strengths": [50, 100], "dose": (50, 100), "freq": ["OD"], "max_day": 100,
                    "timing": "with or without food", "for": "diabetes", "note": "Reduce dose if eGFR < 45."},
    "Empagliflozin": {"class": "diabetes_medication", "strengths": [10, 25], "dose": (10, 25), "freq": ["OD"], "max_day": 25,
                      "timing": "morning", "for": "diabetes", "note": "Preferred with CVD/CKD/heart failure. Genital hygiene, hydration."},
    "Telmisartan": {"class": "antihypertensive", "strengths": [20, 40, 80], "dose": (20, 80), "freq": ["OD"], "max_day": 80,
                    "timing": "morning", "for": "hypertension", "note": "Check potassium and creatinine in 2-4 weeks. Avoid in pregnancy."},
    "Losartan": {"class": "antihypertensive", "strengths": [25, 50, 100], "dose": (25, 100), "freq": ["OD", "BD"], "max_day": 100,
                 "timing": "morning", "for": "hypertension", "note": "Check potassium and creatinine in 2-4 weeks. Avoid in pregnancy."},
    "Amlodipine": {"class": "antihypertensive", "strengths": [2.5, 5, 10], "dose": (2.5, 10), "freq": ["OD"], "max_day": 10,
                   "timing": "morning", "for": "hypertension", "note": "May cause ankle oedema."},
    "Chlorthalidone": {"class": "antihypertensive", "strengths": [12.5, 25], "dose": (12.5, 25), "freq": ["OD"], "max_day": 25,
                       "timing": "morning", "for": "hypertension", "note": "Monitor sodium/potassium."},
    "Atorvastatin": {"class": "lipid_lowering", "strengths": [10, 20, 40, 80], "dose": (10, 80), "freq": ["HS", "OD-N"], "max_day": 80,
                     "timing": "after dinner", "for": "lipid", "note": "Baseline lipid profile and LFT."},
    "Rosuvastatin": {"class": "lipid_lowering", "strengths": [5, 10, 20], "dose": (5, 20), "freq": ["HS", "OD-N"], "max_day": 20,
                     "timing": "after dinner", "for": "lipid", "note": "Baseline lipid profile and LFT."},
    "Aspirin": {"class": "antiplatelet", "strengths": [75, 150], "dose": (75, 150), "freq": ["OD"], "max_day": 150,
                "timing": "after lunch", "for": "secondary_prevention", "note": "Secondary prevention only (established CVD/stroke)."},
}


# ------------------------------------------------------------------ clinical context
def context(form: dict, nlp: dict, diabetes: dict, hypertension: dict, values: dict) -> dict:
    f = nlp.get("features", {})
    text = " ".join(str(form.get(k) or "") for k in ["complaints", "existing_illness", "medications", "family_history"])
    allergies = re.findall(r"allerg\w*\s+(?:to\s+)?([A-Za-z\- ]{3,30})", text, re.I)
    field = (form.get("allergies") or "").strip()
    if field and not re.fullmatch(r"(nkda|nil|none|no|na|n/a|no known (drug )?allerg\w*)\.?", field, re.I):
        allergies += [a for a in re.split(r"[,;/]|\band\b", field) if a.strip()]
    sex, age = nlp.get("sex"), values.get("age")
    labs = diabetes.get("labs_from_text") or {}
    preg_field = (form.get("pregnancy") or "").lower()
    pregnant = preg_field == "yes" or (preg_field != "no" and bool(re.search(r"\bpregnan\w*|\bantenatal\b|\bG\dP\d", text, re.I)))
    return {
        "age": age, "sex": sex,
        "diabetes_eligible": diabetes["level"] == "known" or bool(diabetes.get("criteria_met")),
        "diabetes_known": diabetes["level"] == "known",
        "on_diabetes_med": "diabetes_medication" in (nlp.get("medication_classes") or []),
        "htn_known": hypertension["level"] == "known",
        "on_antihtn": "antihypertensive" in (nlp.get("medication_classes") or []),
        "on_statin": "lipid_lowering" in (nlp.get("medication_classes") or []),
        "on_antiplatelet": "antiplatelet" in (nlp.get("medication_classes") or []),
        "sbp": values.get("systolic_bp"), "dbp": values.get("diastolic_bp"),
        "grbs": values.get("grbs"), "hba1c": labs.get("hba1c_percent"),
        "cvd_or_stroke": f.get("cardiovascular_disease") == 1 or f.get("stroke") == 1,
        "ckd": "kidney_disease" in (nlp.get("conditions") or []),
        "pregnant": pregnant,
        "pregnancy_status": preg_field or "not recorded",
        # explicitly "not pregnant" -> ARB allowed with contraception counselling; otherwise avoid
        "possible_childbearing": sex == "female" and (age or 99) < 50 and preg_field != "no",
        "female_under_50": sex == "female" and (age or 99) < 50,
        "egfr": labs.get("egfr"),
        "allergies": [a.strip().lower() for a in allergies],
        "allergy_field": field or "not recorded",
        "existing_medications": nlp.get("medications") or [],
    }


def _contra(drug: str, c: dict) -> str | None:
    cls_words = {"Atorvastatin": ["statin"], "Rosuvastatin": ["statin"], "Telmisartan": ["arb", "sartan"],
                 "Losartan": ["arb", "sartan"], "Aspirin": ["nsaid", "salicylate"], "Glimepiride": ["sulfonylurea"]}
    for a in c["allergies"]:
        if drug.lower() in a or a.startswith(drug.lower()[:5]) or any(w in a.split() or w in a for w in cls_words.get(drug, [])):
            return f"documented allergy ({a})"
    if drug in ("Telmisartan", "Losartan") and (c["pregnant"] or c["possible_childbearing"]):
        return "ARB avoided: pregnancy / possible pregnancy in a woman of childbearing age (teratogenic) - use amlodipine instead"
    if drug in ("Atorvastatin", "Rosuvastatin") and c["pregnant"]:
        return "statins contraindicated in pregnancy"
    if c["pregnant"] and FORMULARY[drug]["for"] in ("diabetes", "hypertension"):
        return "pregnancy - prescribe under obstetric medicine (insulin for glucose; labetalol / nifedipine for BP)"
    egfr = c.get("egfr")
    if drug == "Metformin":
        if egfr is not None and egfr < 30:
            return f"eGFR {egfr:g} < 30 - metformin contraindicated"
        if egfr is not None and egfr < 45:
            return f"eGFR {egfr:g} (30-45) - ADA: do not initiate metformin"
        if egfr is None and c["ckd"]:
            return "CKD documented but eGFR not recorded - check eGFR first (avoid if < 30)"
    if drug == "Empagliflozin" and egfr is not None and egfr < 20:
        return f"eGFR {egfr:g} < 20 - do not initiate SGLT2 inhibitor"
    if drug == "Sitagliptin" and egfr is not None and egfr < 45:
        return f"eGFR {egfr:g} < 45 - sitagliptin needs dose reduction (50 mg or 25 mg); prescriber to adjust"
    if drug == "Glimepiride" and (c["age"] or 0) >= 65:
        return "sulfonylurea avoided in age >= 65 (hypoglycaemia risk)"
    if drug == "Aspirin" and not c["cvd_or_stroke"]:
        return "aspirin only for secondary prevention (no CVD/stroke documented)"
    return None


def _eligible(drug: str, c: dict) -> str | None:
    """Return a reason if the indication for this drug is NOT established."""
    ind = FORMULARY[drug]["for"]
    if ind == "diabetes" and not c["diabetes_eligible"]:
        return "diabetes not documented / not confirmed - no antidiabetic drug from a risk score alone"
    if ind == "hypertension":
        bp_high = c["sbp"] is not None and (c["sbp"] >= 160 or c["dbp"] >= 100)
        if not (c["htn_known"] or bp_high):
            return "hypertension not established - confirm with repeat readings before starting drugs"
    if ind == "lipid" and not (c["cvd_or_stroke"] or (c["diabetes_eligible"] and 40 <= (c["age"] or 0) <= 75)):
        return "statin indication not established (needs diabetes age 40-75 or CVD)"
    return None


# ------------------------------------------------------------------ rule-based draft (fallback)
def rule_draft(c: dict, skipped: list | None = None) -> list[dict]:
    items = []
    for m in c["existing_medications"]:
        if not m.get("text"):
            continue
        items.append({"drug": m.get("text"), "dose": m.get("dose") or "as before", "frequency": m.get("frequency") or "as before",
                      "timing": "", "duration": "continue", "indication": ", ".join(m.get("classes") or []) or "existing",
                      "action": "continue"})

    def add(drug, mg, freq, duration, why) -> bool:
        reason = _contra(drug, c) or _eligible(drug, c)
        if reason:
            if skipped is not None and FORMULARY[drug]["for"] != "secondary_prevention":
                skipped.append(f"{drug}: {reason}")
            return False
        items.append({"drug": drug, "dose": f"{mg:g} mg", "frequency": freq, "timing": FORMULARY[drug]["timing"],
                      "duration": duration, "indication": why, "action": "new"})
        return True

    if c["diabetes_eligible"] and not c["on_diabetes_med"]:
        started = add("Metformin", 500, "OD", "1 week, then 500 mg BD (review at 2 weeks)", "Type 2 diabetes - first line")
        egfr_low = c.get("egfr") is not None and c["egfr"] < 45
        if c["cvd_or_stroke"] or (not started and egfr_low):
            # ADA: SGLT2 inhibitor for CVD, or as the agent of choice when CKD rules out metformin
            add("Empagliflozin", 10, "OD", "30 days",
                "Diabetes with established CVD (cardio-protective)" if c["cvd_or_stroke"] else
                f"Diabetes with reduced eGFR ({c['egfr']:g}) - metformin not started; kidney-protective")
    if not c["on_antihtn"] and c["sbp"] is not None and (c["htn_known"] or c["sbp"] >= 160 or c["dbp"] >= 100):
        if c["diabetes_eligible"] and not (c["pregnant"] or c["possible_childbearing"]):
            add("Telmisartan", 40, "OD", "30 days", "Hypertension with diabetes (ARB preferred)")
        else:
            add("Amlodipine", 5, "OD", "30 days", "Hypertension")
    if not c["on_statin"]:
        if c["cvd_or_stroke"]:
            add("Atorvastatin", 40, "HS", "30 days", "Secondary prevention (high-intensity statin)")
        elif c["diabetes_eligible"] and 40 <= (c["age"] or 0) <= 75:
            add("Atorvastatin", 10, "HS", "30 days", "Diabetes age 40-75 (primary prevention)")
    if c["cvd_or_stroke"] and not c["on_antiplatelet"]:
        add("Aspirin", 75, "OD", "30 days", "Secondary prevention")
    return items


# ------------------------------------------------------------------ validation of LLM items
def _mg(dose: str) -> float | None:
    m = re.search(r"(\d+(\.\d+)?)\s*mg", str(dose or ""), re.I)
    return float(m.group(1)) if m else None


def validate(items: list[dict], c: dict) -> tuple[list[dict], list[str]]:
    ok, removed = [], []
    existing = {str(m.get("text") or "").lower() for m in c["existing_medications"]}
    for it in items:
        name = str(it.get("drug") or "").strip()
        if it.get("action") == "continue":
            if any(name.lower()[:6] in e for e in existing):
                ok.append(it)
            else:
                removed.append(f"{name}: marked 'continue' but not in the recorded medication list")
            continue
        key = next((k for k in FORMULARY if k.lower() in name.lower()), None)
        if not key:
            removed.append(f"{name}: not in the approved formulary")
            continue
        spec, mg = FORMULARY[key], _mg(it.get("dose"))
        freq = str(it.get("frequency") or "").upper().replace(" ", "")
        problems = []
        if mg is None or not (spec["dose"][0] <= mg <= spec["dose"][1]):
            problems.append(f"dose {it.get('dose')} outside {spec['dose'][0]:g}-{spec['dose'][1]:g} mg")
        if freq not in spec["freq"]:
            problems.append(f"frequency {it.get('frequency')} not allowed ({'/'.join(spec['freq'])})")
        elif mg and mg * (2 if freq == "BD" else 3 if freq == "TDS" else 1) > spec["max_day"]:
            problems.append("exceeds maximum daily dose")
        reason = _contra(key, c) or _eligible(key, c)
        if reason:
            problems.append(reason)
        if any(key.lower() in e for e in existing):
            problems.append("patient already takes this drug")
        if problems:
            removed.append(f"{key}: " + "; ".join(problems))
            continue
        it = {**it, "drug": key, "dose": f"{mg:g} mg", "frequency": freq, "timing": it.get("timing") or spec["timing"]}
        ok.append(it)
    return ok, removed


# ------------------------------------------------------------------ public
PRESCRIBE_SYSTEM = """You are a consultant physician writing a prescription for a screened patient.
Follow ADA Standards of Care and ACC/AHA BP guidelines. Strict rules:
- Prescribe a drug ONLY for a condition that is documented or meets diagnostic criteria (flags given).
  If diabetes is only "risk" (not eligible), prescribe NO antidiabetic drug - give lifestyle advice and tests.
- Start hypertension drugs only if hypertension is known OR BP >= 160/100; otherwise advise repeat readings.
- Continue the patient's existing medications (action "continue", copy name/dose/frequency as recorded)
  unless there is a clear problem.
- Use ONLY drugs from this formulary, with doses inside the given range and allowed frequencies:
FORMULARY
- Frequency must be one of OD, OD-N, HS, BD, TDS. Dose as "<number> mg" per administration.
Return ONLY JSON:
{"items":[{"drug":str,"dose":str,"frequency":str,"timing":str,"duration":str,"indication":str,"action":"new"|"continue"}],
 "non_drug_advice":[str], "investigations":[str], "review_after":str, "notes_for_prescriber":[str]}"""


def draft(form, nlp, diabetes, hypertension, values, use_llm=True) -> dict:
    from . import llm
    c = context(form, nlp, diabetes, hypertension, values)
    source, extra, removed = "Rule-based guideline engine", {}, []
    items = None
    if use_llm and llm.available():
        try:
            formulary = {k: {kk: v[kk] for kk in ("class", "dose", "freq", "max_day", "timing", "for")} for k, v in FORMULARY.items()}
            sysmsg = PRESCRIBE_SYSTEM.replace("FORMULARY", json.dumps(formulary))
            data = json.loads(llm._chat(sysmsg, json.dumps({
                "patient": {k: c[k] for k in ("age", "sex", "pregnant", "pregnancy_status", "sbp", "dbp", "grbs", "hba1c",
                                                     "egfr", "ckd", "allergies", "allergy_field")},
                "diabetes_status": diabetes["status"], "diabetes_eligible_for_drugs": c["diabetes_eligible"],
                "hypertension_status": hypertension["status"], "hypertension_known": c["htn_known"],
                "cvd_or_stroke": c["cvd_or_stroke"], "existing_medications": c["existing_medications"],
                "conditions": nlp.get("conditions"), "symptoms": nlp.get("symptoms"),
                "clinician_advice": form.get("advice"), "follow_up": form.get("follow_up")}, default=str), json_mode=True))
            items, removed = validate(data.get("items") or [], c)
            # Guideline floor: if the guideline engine says an indication needs treatment and the
            # LLM's (safety-checked) list has nothing for it - because the LLM omitted it or the
            # safety check removed its choice - add the engine's safe choice, clearly labelled.
            covered = {FORMULARY[i["drug"]]["for"] for i in items if i.get("drug") in FORMULARY}
            covered |= {FORMULARY[k]["for"] for k in FORMULARY for m in c["existing_medications"]
                        if k.lower() in str(m.get("text") or "").lower()}
            for alt in rule_draft(c):
                ind = FORMULARY.get(alt["drug"], {}).get("for")
                if alt["action"] == "new" and ind and ind not in covered:
                    alt["indication"] += " (added by guideline engine)"
                    items.append(alt)
                    covered.add(ind)
            extra = {k: data.get(k) for k in ("non_drug_advice", "investigations", "review_after", "notes_for_prescriber")}
            source = f"LLM ({llm.LLM_MODEL}) + formulary safety check"
        except Exception as e:  # noqa: BLE001
            removed.append(f"LLM prescription unavailable ({type(e).__name__}); rule-based draft used")
            items = None
    if items is None:
        items = rule_draft(c, removed)
    for it in items:
        code = str(it.get("frequency") or "").upper()
        it["pattern"], it["frequency_text"] = FREQ.get(code, (it.get("frequency"), it.get("frequency")))
        spec = FORMULARY.get(it.get("drug"))
        it["counselling"] = spec["note"] if spec else ""
    if not extra.get("investigations"):
        extra["investigations"] = ["Fasting plasma glucose", "HbA1c", "Lipid profile", "Serum creatinine / eGFR", "Urine albumin-creatinine ratio"]
    if not extra.get("review_after"):
        extra["review_after"] = "2 weeks" if any(i["action"] == "new" for i in items) else "3 months"
    warnings = []
    if not c["diabetes_eligible"] and diabetes["level"] in ("high", "moderate"):
        warnings.append("Elevated diabetes risk but not confirmed: no antidiabetic drug prescribed - confirm with FPG/HbA1c first.")
    if c["sbp"] is not None and c["sbp"] >= 140 and not c["htn_known"] and c["sbp"] < 160 and (c["dbp"] or 0) < 100:
        warnings.append("Raised BP on a single reading: confirm with repeat/home readings before starting antihypertensives.")
    renal = {"Metformin", "Empagliflozin", "Sitagliptin", "Telmisartan", "Losartan"}
    new_renal = sorted({it["drug"] for it in items if it.get("action") == "new" and it.get("drug") in renal})
    if new_renal and c.get("egfr") is None:
        warnings.append(f"Kidney function not recorded: check serum creatinine/eGFR (and potassium for ARBs) before starting {', '.join(new_renal)}.")
    if c["female_under_50"] and not c["pregnant"] and c["pregnancy_status"] == "no" and \
            any(it.get("drug") in ("Telmisartan", "Losartan") for it in items):
        warnings.append("Woman of childbearing age on an ARB: counsel on effective contraception; stop if pregnancy is planned or occurs.")
    if c["pregnant"] and c["sbp"] is not None and (c["sbp"] >= 160 or (c["dbp"] or 0) >= 110):
        warnings.append(f"URGENT: severe hypertension in pregnancy ({c['sbp']:g}/{c['dbp']:g}) - same-day obstetric assessment.")
    if c["pregnant"]:
        warnings.append("Pregnant patient: diabetes/hypertension in pregnancy need obstetric-medicine care; oral agents limited.")
    if c["allergy_field"] == "not recorded":
        warnings.append("Drug allergy status not recorded - confirm before dispensing.")
    if diabetes.get("hybrid_score") is not None and (c["grbs"] or 0) >= 300:
        warnings.append("GRBS >= 300 mg/dL: assess for symptoms/ketones; consider insulin and specialist referral.")
    return {"status": "DRAFT - requires clinician review and signature", "source": source, "items": items,
            "removed_by_safety_check": removed, "warnings": warnings, **extra}


def as_text(rx: dict) -> str:
    L = ["PRESCRIPTION (DRAFT - not valid until reviewed and signed by the treating clinician):", "Rx"]
    if not rx["items"]:
        L.append("  No medication prescribed at this visit.")
    for i, it in enumerate(rx["items"], 1):
        if it["action"] == "continue":
            L.append(f"  {i}. {it['drug']} - continue as before  [continue; {it.get('indication') or 'existing'}]")
            continue
        L.append(f"  {i}. {'Tab ' if it['action'] == 'new' and not str(it['drug']).lower().startswith(('tab', 'cap', 'inj')) else ''}{it['drug']} "
                 f"{it['dose']} - {it['pattern']} ({it['frequency_text']}), {it.get('timing') or ''} - {it.get('duration') or ''}"
                 f"  [{it['action']}; {it.get('indication') or ''}]")
    for w in rx.get("warnings", []):
        L.append(f"  ! {w}")
    if rx.get("non_drug_advice"):
        L += ["Advice:"] + [f"  - {a}" for a in rx["non_drug_advice"]]
    L += ["Investigations:"] + [f"  - {a}" for a in rx["investigations"]]
    L.append(f"Review after: {rx['review_after']}")
    return "\n".join(L)
