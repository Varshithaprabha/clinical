"""Rule-based clinical NLP extraction (NegEx-style negation handling).

Works on the free-text sections of the clinician form and returns structured flags used by
the hybrid model. The LLM extractor (src/llm.py) produces the same schema; this module is
the deterministic baseline and the fallback when no LLM is reachable.
"""
from __future__ import annotations

import re

CONDITIONS = {
    "diabetes": r"\b(t[12]\s?dm|dm\s?(type\s?)?(1|2|i|ii)?|diabet\w*|high (blood )?sugar|sugar (problem|disease)|madhumeha)\b",
    "hypertension": r"\b(htn|hypertens\w*|high (blood pressure|bp)|raised bp|elevated bp|high pressure)\b",
    "cardiovascular_disease": r"\b(ihd|cad|cvd|mi|myocardial infarction|heart attack|angina|coronary\w*( artery disease)?|heart (disease|failure)|ischa?emic heart|cabg|ptca|stent)\b",
    "stroke": r"\b(stroke|cva|tia|cerebrovascular accident|hemipar\w+|paralysis)\b",
    "thyroid": r"\b(thyroid\w*|hypothyroid\w*|hyperthyroid\w*|thyroxine|levothyroxine|eltroxin)\b",
    "cancer": r"\b(CA\b|cancer|carcinoma|malignan\w+|tumou?r|lymphoma|leuka?emia)",
    "kidney_disease": r"\b(ckd|chronic kidney|renal (failure|disease)|nephropathy)\b",
    "dyslipidemia": r"\b(dyslipid\w*|hyperlipid\w*|high cholesterol|hypercholest\w*)\b",
}

SYMPTOMS = {
    "polyuria": r"\b(polyuria|frequent urination|passing urine (frequently|often)|nocturia|urinat\w+ (a lot|frequently))\b",
    "polydipsia": r"\b(polydipsia|excessive thirst|increased thirst|very thirsty|thirst\w*)\b",
    "polyphagia": r"\b(polyphagia|excessive hunger|increased (appetite|hunger))\b",
    "weight_loss": r"\b(weight loss|losing weight|lost weight|unintentional weight)\b",
    "fatigue": r"\b(fatigue|tired\w*|weakness|lethargy)\b",
    "blurred_vision": r"\b(blurr?(ed|ing)? vision|vision (blurr\w*|problem)|diminished vision)\b",
    "numbness_tingling": r"\b(numbness|tingling|burning feet|pins and needles|neuropath\w*)\b",
    "slow_healing": r"\b(slow(ly)?[- ]healing|non[- ]healing|ulcer)\b",
    "headache": r"\b(headache|giddiness|dizziness)\b",
    "chest_pain": r"\b(chest pain|palpitation\w*|breathless\w*|dyspno?ea)\b",
}

MEDICATIONS = {
    "diabetes_medication": r"\b(metformin|glycomet|glimepiride|gliclazide|glibenclamide|glipizide|insulin|sitagliptin|vildagliptin|linagliptin|teneligliptin|empagliflozin|dapagliflozin|canagliflozin|pioglitazone|acarbose|voglibose|semaglutide|liraglutide|dulaglutide|ohas?|oral hypoglyc\w*)\b",
    "antihypertensive": r"\b(amlodipine|telmisartan|losartan|olmesartan|valsartan|enalapril|ramipril|lisinopril|atenolol|metoprolol|bisoprolol|nebivolol|hydrochlorothiazide|hctz|chlorthalidone|indapamide|cilnidipine|nifedipine|prazosin|anti-?hypertensive\w*)\b",
    "lipid_lowering": r"\b(atorvastatin|rosuvastatin|simvastatin|statin\w*|fenofibrate|ezetimibe)\b",
    "antiplatelet": r"\b(aspirin|ecosprin|clopidogrel|ticagrelor|prasugrel)\b",
    "thyroid_medication": r"\b(levothyroxine|thyroxine|eltroxin|thyronorm|carbimazole|methimazole)\b",
}

FAMILY_RELATIONS = r"\b(mother|father|mom|dad|parent\w*|brother|sister|sibling\w*|son|daughter|uncle|aunt|grand\w+|maternal|paternal|family)\b"
NEGATION_CUES = r"\b(no|not|nil|none|denies|denied|negative for|without|absent|never|free of|rules? out|ruled out|unremarkable|nothing significant|insignificant|not significant)\b"
FREQ = r"\b(od|bd|bid|tds|tid|qid|hs|sos|once daily|twice daily|thrice daily|three times (a|per) day|daily|weekly)\b"


def _clauses(text: str) -> list[str]:
    return [c.strip() for c in re.split(r"(?<!\d)\.|\.(?!\d)|[;\n]|,(?=\s*(?:but|however)\b)|\bbut\b|\bhowever\b", text or "", flags=re.I) if c.strip()]


def _find(pattern: str, clause: str, case_sensitive: bool = False) -> list[re.Match]:
    return list(re.finditer(pattern, clause, 0 if case_sensitive else re.I))


POST_NEGATION = r"^\W*(\w+\W+){0,3}?(ruled out|excluded|is negative|negative|absent|not present|unlikely|denied)\b"


def _negated(clause: str, start: int, end: int | None = None) -> bool:
    """NegEx-lite: a negation cue within the 7 words before the term (pre-negation) or a
    post-negation phrase right after it ("thyroid disease ruled out"), in the same clause."""
    if end is not None and re.search(POST_NEGATION, clause[end:], re.I):
        return True
    words = [w for w in re.split(r"[\s/,]+", clause[:start]) if re.search(r"\w", w)]
    window = " ".join(words[-7:])
    return re.search(NEGATION_CUES, window, re.I) is not None


def _scan(text: str, lexicon: dict) -> dict:
    found, negated, evidence = set(), set(), {}
    for clause in _clauses(text):
        for name, pat in lexicon.items():
            cs = name == "cancer"  # "CA" must be upper-case to avoid matching "ca" in words
            for m in _find(pat, clause, cs):
                if _negated(clause, m.start(), m.end()):
                    negated.add(name)
                else:
                    found.add(name)
                    evidence.setdefault(name, clause)
    return {"present": sorted(found), "negated": sorted(negated - found), "evidence": evidence}


def _labs(text: str) -> dict:
    t = text or ""
    out = {}
    m = re.search(r"\b(?:hba1c|a1c|glycated ha?emoglobin)\D{0,15}?(?P<v>\d{1,2}(?:\.\d+)?)", t, re.I)
    if m:
        out["hba1c_percent"] = float(m.group("v"))
    m = re.search(r"\b(?:fbs|fpg|fbg|fasting (?:blood |plasma )?(?:glucose|sugar))\D{0,15}?(?P<v>\d{1,3}(?:\.\d+)?)(?P<u>\s*mmol)?", t, re.I)
    if m:
        v = float(m.group("v"))
        out["fasting_glucose_mgdl"] = round(v * 18.016) if (m.group("u") or v < 35) else v
    m = re.search(r"\b(ppbs|pp(bg)?|post[- ]?prandial( blood)?( glucose| sugar)?)\D{0,15}(\d{2,3}(\.\d+)?)", t, re.I)
    if m:
        out["postprandial_glucose_mgdl"] = float(m.group(5))
    return out


def _medication_list(text: str) -> list[dict]:
    meds = []
    if not text or re.fullmatch(r"\s*(nil|none|no|na|n/a|-)\s*\.?\s*", text, re.I):
        return meds
    for part in re.split(r"[,;\n]|\band\b|\+", text):
        part = part.strip()
        if not part:
            continue
        cls = [n for n, p in MEDICATIONS.items() if re.search(p, part, re.I)]
        dose = re.search(r"(\d+(\.\d+)?\s?(mg|mcg|g|iu|units?|ml))", part, re.I)
        freq = re.search(FREQ, part, re.I)
        meds.append({"text": part, "classes": cls,
                     "dose": dose.group(1) if dose else None,
                     "frequency": freq.group(1) if freq else None})
    return meds


def extract(complaints: str = "", existing_illness: str = "", medications: str = "",
            family_history: str = "", other_text: str = "") -> dict:
    """Return structured, negation-aware findings from the clinician free-text fields."""
    personal_text = " . ".join([complaints or "", existing_illness or "", other_text or ""])
    personal = _scan(personal_text, CONDITIONS)
    symptoms = _scan(" . ".join([complaints or "", other_text or ""]), SYMPTOMS)
    meds = _scan(medications or "", MEDICATIONS)
    med_list = _medication_list(medications or "")

    # Family history field: every non-negated condition is a family finding; the relation
    # word (mother, father, ...) is captured when present
    fam_found, fam_neg, fam_ev = set(), set(), {}
    for clause in _clauses(family_history or ""):
        for name, pat in CONDITIONS.items():
            for m in _find(pat, clause, name == "cancer"):
                if _negated(clause, m.start(), m.end()):
                    fam_neg.add(name)
                else:
                    fam_found.add(name)
                    rel = re.search(FAMILY_RELATIONS, clause, re.I)
                    fam_ev[name] = {"relation": rel.group(1) if rel else "unspecified", "evidence": clause}

    p = set(personal["present"])
    mc = set(meds["present"])
    sex = None
    all_text = " ".join([complaints or "", existing_illness or "", other_text or ""])
    if re.search(r"\b(female|woman|lady|mrs|ms|pregnan\w*|g\dp\d)\b", all_text, re.I):
        sex = "female"
    elif re.search(r"\b(male|man|gentleman|mr)\b", all_text, re.I):
        sex = "male"

    def provided(*fields):
        return any((f or "").strip() for f in fields)

    hist_given = provided(existing_illness, complaints, medications)
    fam_given = provided(family_history)

    def flag(cond_present: bool, given: bool):
        # 1 = documented, 0 = section filled but not mentioned / negated, None = section empty
        return 1.0 if cond_present else (0.0 if given else None)

    features = {
        "known_diabetes": flag("diabetes" in p or "diabetes_medication" in mc, hist_given),
        "on_diabetes_medication": flag("diabetes_medication" in mc, provided(medications)),
        "hypertensive": flag("hypertension" in p or "antihypertensive" in mc, hist_given),
        "cardiovascular_disease": flag("cardiovascular_disease" in p, hist_given),
        "stroke": flag("stroke" in p, hist_given),
        "family_diabetes": flag("diabetes" in fam_found, fam_given),
        "family_hypertension": flag("hypertension" in fam_found, fam_given),
        "symptom_count": float(len(symptoms["present"])),
        "classic_hyperglycaemia_symptoms": float(len({"polyuria", "polydipsia", "polyphagia", "weight_loss"} & set(symptoms["present"]))),
    }
    return {
        "method": "rule-based (NegEx-style)",
        "conditions": personal["present"],
        "negated_conditions": personal["negated"],
        "condition_evidence": personal["evidence"],
        "symptoms": symptoms["present"],
        "negated_symptoms": symptoms["negated"],
        "medication_classes": meds["present"],
        "medications": med_list,
        "family_history": [{"condition": k, **v} for k, v in fam_ev.items()],
        "family_negated": sorted(fam_neg - fam_found),
        "labs": _labs(" ".join([complaints or "", existing_illness or "", other_text or "", medications or ""])),
        "sex": sex,
        "features": features,
    }
