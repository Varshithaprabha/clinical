"""End-to-end assessment of one clinician-entered screening record."""
from __future__ import annotations

import re
import traceback

import joblib
import numpy as np
import pandas as pd

from . import llm, nlp_extract, notes, prescription, text_models
from .config import MODELS_DIR
from .preprocess import PLAUSIBLE

_M: dict = {}

FRIENDLY = {
    "age": "Age", "gender_male": "Sex (male)", "bmi": "BMI", "glucose": "GRBS (blood glucose)",
    "systolic_bp": "Systolic BP", "diastolic_bp": "Diastolic BP", "hypertensive": "Hypertension history",
    "cardiovascular_disease": "Cardiovascular disease", "stroke": "Stroke history",
    "family_diabetes": "Family history of diabetes", "family_hypertension": "Family history of hypertension",
    "symptom_count": "Number of symptoms reported", "biobert_text": "Clinical text (BioBERT context)",
}


def models() -> dict:
    if not _M:
        _M["general"] = joblib.load(MODELS_DIR / "general_structured.joblib")
        _M["hybrid"] = joblib.load(MODELS_DIR / "hybrid_xgb.joblib")
        _M["htn"] = joblib.load(MODELS_DIR / "hypertension_xgb.joblib")
        _M["tfidf"] = joblib.load(MODELS_DIR / "tfidf_svm.joblib")
        _M["biobert"] = joblib.load(MODELS_DIR / "biobert_clf.joblib")
    return _M


# ------------------------------------------------------------------ validation
def _num(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return "invalid"


def validate(form: dict) -> dict:
    issues, values = [], {}
    spec = {"age": ("Age", "age", 1), "weight": ("Weight (kg)", "weight", 1),
            "height": ("Height (cm)", "height_m", 0.01), "grbs": ("GRBS (mg/dL)", "glucose_mgdl", 1)}
    for key, (label, pkey, scale) in spec.items():
        v = _num(form.get(key))
        if key == "height" and isinstance(v, float) and 1.0 <= v <= 2.5:  # typed in metres
            issues.append({"field": label, "level": "info", "message": f"{v:g} read as metres - converted to {v * 100:g} cm"})
            v = round(v * 100, 1)
        if v is None:
            values[key] = None
            issues.append({"field": label, "level": "missing", "message": "Not recorded"})
        elif v == "invalid":
            values[key] = None
            issues.append({"field": label, "level": "error", "message": f"'{form.get(key)}' is not a number - treated as missing"})
        else:
            lo, hi = PLAUSIBLE[pkey]
            if not (lo <= v * scale <= hi):
                values[key] = None
                issues.append({"field": label, "level": "error",
                               "message": f"{v:g} is outside the plausible range ({lo / scale:g}-{hi / scale:g}) - treated as missing, please re-check"})
            else:
                values[key] = v

    sbp = dbp = None
    bp = (form.get("bp") or "").strip()
    if not bp:
        issues.append({"field": "BP (mmHg)", "level": "missing", "message": "Not recorded"})
    else:
        m = re.fullmatch(r"\s*(\d{2,3})\s*[/\\-]\s*(\d{2,3})\s*", bp)
        if not m:
            issues.append({"field": "BP (mmHg)", "level": "error", "message": f"'{bp}' is not in systolic/diastolic format (e.g. 120/80)"})
        else:
            s, d = float(m.group(1)), float(m.group(2))
            ok = PLAUSIBLE["systolic_bp"][0] <= s <= PLAUSIBLE["systolic_bp"][1] and \
                PLAUSIBLE["diastolic_bp"][0] <= d <= PLAUSIBLE["diastolic_bp"][1] and d < s
            if ok:
                sbp, dbp = s, d
            else:
                issues.append({"field": "BP (mmHg)", "level": "error", "message": f"{bp} is implausible (check order / values) - treated as missing"})
    values["systolic_bp"], values["diastolic_bp"] = sbp, dbp

    bmi = None
    if values.get("weight") and values.get("height"):
        bmi = round(values["weight"] / (values["height"] / 100) ** 2, 1)
        if not (PLAUSIBLE["bmi"][0] <= bmi <= PLAUSIBLE["bmi"][1]):
            issues.append({"field": "BMI", "level": "error", "message": f"Computed BMI {bmi} is implausible - check height/weight"})
            bmi = None
    values["bmi"] = bmi
    # optional laboratory fields: validated if entered, never required
    for key, label, lo, hi in [("fbs", "Fasting glucose (mg/dL)", 40, 600), ("hba1c", "HbA1c (%)", 3, 20),
                               ("egfr", "eGFR (mL/min/1.73m2)", 2, 150)]:
        x = _num(form.get(key))
        values[key] = None
        if x == "invalid":
            issues.append({"field": label, "level": "error", "message": f"'{form.get(key)}' is not a number - ignored"})
        elif x is not None and key == "fbs" and x < 35:  # entered in mmol/L
            values[key] = round(x * 18.016)
            issues.append({"field": label, "level": "info", "message": f"{x:g} read as mmol/L - converted to {values[key]} mg/dL"})
        elif x is not None and not (lo <= x <= hi):
            issues.append({"field": label, "level": "error", "message": f"{x:g} is outside the plausible range ({lo}-{hi}) - ignored"})
        elif x is not None:
            values[key] = x
    available = sum(x is not None for x in [values.get("weight"), values.get("height"), sbp, values.get("grbs")])
    return {"values": values, "issues": issues, "available_measurements": available}


def bmi_category(bmi):
    if bmi is None:
        return None
    who = ("Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else
           "Obese class I" if bmi < 35 else "Obese class II" if bmi < 40 else "Obese class III")
    asian = ("Underweight" if bmi < 18.5 else "Normal" if bmi < 23 else "Overweight (at risk)" if bmi < 27.5 else "Obese")
    return {"who": who, "asian": asian}


def bp_category(s, d):
    if s is None or d is None:
        return None
    if s > 180 or d > 120:
        return "Hypertensive crisis"
    if s >= 140 or d >= 90:
        return "Stage 2 hypertension"
    if s >= 130 or d >= 80:
        return "Stage 1 hypertension"
    if s >= 120:
        return "Elevated"
    return "Normal"


# ------------------------------------------------------------------ NLP
def run_nlp(form: dict, use_llm: bool = True) -> dict:
    sections = {k: form.get(k, "") for k in ["complaints", "existing_illness", "medications", "family_history"]}
    other = " . ".join(filter(None, [form.get("vision"), form.get("lower_limbs")]))
    rule = nlp_extract.extract(other_text=other, **sections)
    result, err = rule, None
    if use_llm and llm.available() and any((v or "").strip() for v in list(sections.values()) + [other]):
        try:
            result = llm.extract({**sections, "vision": form.get("vision", ""), "lower_limbs": form.get("lower_limbs", "")})
            # rule-based labs are kept if the LLM missed a number
            result["labs"] = {**rule["labs"], **result["labs"]}
        except Exception as e:  # noqa: BLE001
            err = f"LLM extraction unavailable ({type(e).__name__}); rule-based extractor used"
            result = rule
    result["rule_based_features"] = rule["features"]
    result["note"] = err
    return result


# ------------------------------------------------------------------ explainability
def _explain(hyb: dict, row: pd.DataFrame) -> dict:
    out = {}
    try:
        import shap
        expl = shap.TreeExplainer(hyb["model"])
        sv = expl.shap_values(row)[0]
        contrib = {}
        for c, v in zip(row.columns, sv):
            key = "biobert_text" if c.startswith("biobert_pc") else c
            contrib[key] = contrib.get(key, 0.0) + float(v)
        items = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))
        out["shap"] = [{"feature": FRIENDLY.get(k, k), "value": _fmt(row.iloc[0].get(k)) if k != "biobert_text" else "text",
                        "contribution": round(v, 4)} for k, v in items[:8]]
        out["shap_base_value"] = float(np.ravel(expl.expected_value)[0])
    except Exception as e:  # noqa: BLE001
        out["shap_error"] = str(e)
    try:
        from lime.lime_tabular import LimeTabularExplainer
        bg = hyb["background"]
        med = bg.median(numeric_only=True)
        bgf = bg.fillna(med)
        cols = list(bg.columns)
        explainer = LimeTabularExplainer(bgf.values, feature_names=[FRIENDLY.get(c, c) for c in cols],
                                         class_names=["No diabetes", "Diabetes"], mode="classification",
                                         discretize_continuous=True, random_state=0)
        inst = row.fillna(med).values[0]  # LIME needs a complete point to perturb around; prediction itself used NaN natively
        fn = lambda a: hyb["model"].predict_proba(pd.DataFrame(a, columns=cols))  # noqa: E731
        exp = explainer.explain_instance(inst, fn, num_features=8, num_samples=1500)
        out["lime"] = [{"rule": r, "weight": round(float(w), 4)} for r, w in exp.as_list()
                       if not r.split()[0].startswith("biobert") and "biobert" not in r][:6]
    except Exception as e:  # noqa: BLE001
        out["lime_error"] = str(e)
    return out


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "missing"
    return round(float(v), 1)


# ------------------------------------------------------------------ main
def assess(form: dict, overrides: dict | None = None, use_llm: bool = True) -> dict:
    M = models()
    val = validate(form)
    v = val["values"]
    nlp = run_nlp(form, use_llm=use_llm)
    feats = dict(nlp["features"])
    if overrides:  # clinician-verified corrections from the NLP review card
        for k, x in overrides.items():
            if k in feats:
                feats[k] = None if x is None else float(x)
        nlp["clinician_verified"] = True
    # Sex: the form field is authoritative; free text is only a fallback. "other"/blank -> unknown
    # (the models were trained on male/female, so unknown is passed as missing, not guessed).
    form_sex = (form.get("sex") or "").strip().lower()
    text_sex = nlp.get("sex")
    if form_sex in ("male", "female"):
        sex, sex_source = form_sex, "form"
        if text_sex and text_sex != form_sex:
            val["issues"].append({"field": "Sex", "level": "error",
                                  "message": f"Form says {form_sex} but the notes suggest {text_sex} - form value used, please check"})
    elif form_sex == "other":
        sex, sex_source = None, "form (other / not stated)"
    elif text_sex:
        sex, sex_source = text_sex, "clinical text"
    else:
        sex, sex_source = None, None
        val["issues"].append({"field": "Sex", "level": "missing", "message": "Not recorded - treated as unknown by the models"})
    nlp["sex"], nlp["sex_source"] = sex, sex_source  # prescription safety checks read this
    val["sex"], val["sex_source"] = sex, sex_source
    gender_male = 1.0 if sex == "male" else 0.0 if sex == "female" else np.nan

    def nz(x):
        return np.nan if x is None else float(x)

    labs = dict(nlp.get("labs", {}))
    for fk, lk in [("fbs", "fasting_glucose_mgdl"), ("hba1c", "hba1c_percent"), ("egfr", "egfr")]:
        if v.get(fk) is not None:  # structured form value beats a number found in free text
            labs[lk] = v[fk]
    nlp["labs"] = labs
    base = {"age": nz(v["age"]), "gender_male": gender_male, "bmi": nz(v["bmi"]), "glucose": nz(v["grbs"]),
            "systolic_bp": nz(v["systolic_bp"]), "diastolic_bp": nz(v["diastolic_bp"]),
            "family_diabetes": nz(feats["family_diabetes"]), "hypertensive": nz(feats["hypertensive"]),
            "family_hypertension": nz(feats["family_hypertension"]),
            "cardiovascular_disease": nz(feats["cardiovascular_disease"]), "stroke": nz(feats["stroke"]),
            "symptom_count": nz(feats["symptom_count"])}

    # ---- text models
    sections = {k: form.get(k, "") for k in ["complaints", "existing_illness", "medications", "family_history"]}
    has_text = any((s or "").strip() for s in sections.values())
    text = notes.note_text(sections)
    text_out = {}
    pcs = np.full(len(M["hybrid"]["columns"]) - len(M["hybrid"]["struct"]) - len(M["hybrid"]["nlp"]), np.nan)
    if has_text:
        text_out["TF-IDF + SVM"] = float(M["tfidf"].predict_proba([text])[0, 1])
        try:
            emb = text_models.embed([text])
            text_out["BioBERT + classifier"] = float(M["biobert"].predict_proba(emb)[0, 1])
            pcs = M["hybrid"]["pca"].transform(emb)[0]
        except Exception as e:  # noqa: BLE001
            text_out["biobert_error"] = str(e)

    # ---- hybrid XGBoost (primary diabetes model)
    hyb = M["hybrid"]
    hrow = {c: base.get(c, np.nan) for c in hyb["struct"] + hyb["nlp"]}
    hrow.update({f"biobert_pc{i+1}": float(x) for i, x in enumerate(pcs)})
    hrow = pd.DataFrame([hrow])[hyb["columns"]].astype(float)
    p_h = float(hyb["model"].predict_proba(hrow)[0, 1])

    # ---- general structured models (the five mandatory classifiers)
    G = M["general"]
    grow = pd.DataFrame([{c: base.get(c, np.nan) for c in G["features"]}]).astype(float)
    structured = {}
    for name, mdl in G["models"].items():
        structured[name] = {"probability": round(float(mdl.predict_proba(grow)[0, 1]), 3),
                            "threshold": round(G["thresholds"].get(name, 0.5), 3)}
        structured[name]["flag"] = structured[name]["probability"] >= structured[name]["threshold"]

    # ---- diabetes decision logic with abstention
    grbs, hba1c, fpg = v["grbs"], labs.get("hba1c_percent"), labs.get("fasting_glucose_mgdl")
    classic = (feats.get("classic_hyperglycaemia_symptoms") or 0) > 0
    criteria = []
    if hba1c is not None and hba1c >= 6.5:
        criteria.append(f"HbA1c {hba1c}% (>= 6.5%)")
    if fpg is not None and fpg >= 126:
        criteria.append(f"Fasting glucose {fpg:g} mg/dL (>= 126)")
    if grbs is not None and grbs >= 200 and classic:
        criteria.append(f"Random glucose {grbs:g} mg/dL (>= 200) with classic symptoms")
    n_flag = sum(s["flag"] for s in structured.values()) + int(p_h >= hyb["threshold"])
    n_models = len(structured) + 1
    sufficient = (grbs is not None or hba1c is not None or fpg is not None) and v["age"] is not None
    if feats.get("known_diabetes") == 1:
        dstatus, dlevel = "Known diabetes (documented in history / medication)", "known"
    elif criteria:
        dstatus, dlevel = "Meets ADA diagnostic criterion - confirm with repeat test", "high"
    elif not sufficient:
        dstatus, dlevel = "Insufficient data - a glucose value (GRBS / FPG / HbA1c) and age are required", "insufficient"
    else:
        thr = hyb["threshold"]
        # "High" needs the primary model AND most of the ensemble to agree; disagreement -> "Moderate"
        if p_h >= thr and n_flag * 2 > n_models:
            dlevel = "high"
        elif p_h >= thr / 2 or n_flag * 2 > n_models:
            dlevel = "moderate"
        else:
            dlevel = "low"
        dstatus = {"high": "High risk of diabetes", "moderate": "Moderate risk of diabetes", "low": "Low risk of diabetes"}[dlevel]
    diabetes = {"status": dstatus, "level": dlevel, "hybrid_score": round(p_h, 3), "hybrid_threshold": round(hyb["threshold"], 3),
                "models_flagging": f"{n_flag} of {n_models}", "criteria_met": criteria, "structured_models": structured,
                "text_models": {k: (round(x, 3) if isinstance(x, float) else x) for k, x in text_out.items()},
                "grbs_interpretation": (None if grbs is None else
                                        "Diabetic range (>= 200 mg/dL)" if grbs >= 200 else
                                        "Above normal (140-199 mg/dL)" if grbs >= 140 else "Normal range (< 140 mg/dL)"),
                "labs_from_text": labs}

    # ---- hypertension
    H = M["htn"]
    htn_row = pd.DataFrame([{c: base.get(c, np.nan) for c in H["features"]}]).astype(float)
    cat = bp_category(v["systolic_bp"], v["diastolic_bp"])
    if feats.get("hypertensive") == 1:
        hstatus, hlevel = "Known hypertension (documented history / antihypertensive therapy)", "known"
        p_ht = float(H["model"].predict_proba(htn_row)[0, 1]) if cat else None
    elif cat is None:
        hstatus, hlevel, p_ht = "Insufficient data - blood pressure reading required", "insufficient", None
    else:
        p_ht = float(H["model"].predict_proba(htn_row)[0, 1])
        hlevel = "high" if cat in ("Stage 2 hypertension", "Hypertensive crisis") or p_ht >= H["threshold"] else \
                 "moderate" if cat in ("Stage 1 hypertension", "Elevated") else "low"
        hstatus = f"{cat} on this reading" + (" - confirm with repeat readings" if hlevel != "low" else "")
    hypertension = {"status": hstatus, "level": hlevel, "bp_category": cat,
                    "model_score": None if p_ht is None else round(p_ht, 3), "threshold": round(H["threshold"], 3)}

    explain = _explain(hyb, hrow) if dlevel != "insufficient" else {"note": "Not generated - insufficient data"}

    rx = prescription.draft(form, nlp, diabetes, hypertension, v, use_llm=use_llm)
    result = {
        "validation": val, "bmi": {"value": v["bmi"], **(bmi_category(v["bmi"]) or {})},
        "nlp": nlp, "diabetes": diabetes, "hypertension": hypertension, "explainability": explain,
        "prescription": rx,
    }

    # ---- LLM clinician report (text)
    payload = {
        "age": v["age"], "sex": sex, "measurements": {"weight_kg": v["weight"], "height_cm": v["height"], "bp": form.get("bp"),
                                                       "grbs_mgdl": v["grbs"], "bmi": result["bmi"]},
        "clinician_text": {**sections, "vision": form.get("vision"), "lower_limbs": form.get("lower_limbs"),
                           "advice": form.get("advice"), "follow_up": form.get("follow_up")},
        "data_issues": val["issues"], "nlp_extraction": {k: nlp.get(k) for k in ["conditions", "negated_conditions", "symptoms", "medications",
                                                                                 "family_history", "labs", "other_findings"]},
        "diabetes": diabetes, "hypertension": hypertension,
        "explainability": {k: explain.get(k) for k in ["shap", "lime"]},
        "draft_prescription": {k: rx.get(k) for k in ["items", "warnings", "removed_by_safety_check", "review_after"]},
    }
    report, report_src = None, None
    if use_llm and llm.available():
        try:
            report, report_src = llm.generate_report(payload), f"LLM ({llm.LLM_MODEL})"
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            report_src = f"Template (LLM unavailable: {type(e).__name__})"
    if report is None:
        report = template_report(payload)
        report_src = report_src or "Template (LLM not configured)"
    result["report"] = {"text": report.rstrip() + "\n\n" + prescription.as_text(rx), "source": report_src}
    return result


def template_report(p: dict) -> str:
    d, h, m = p["diabetes"], p["hypertension"], p["measurements"]
    L = ["CLINICAL SUMMARY:",
         f"Age {p['age'] or 'not recorded'}, sex {p['sex'] or 'not recorded'}. Weight {m['weight_kg'] or '-'} kg, height {m['height_cm'] or '-'} cm, "
         f"BP {m['bp'] or 'not recorded'}, GRBS {m['grbs_mgdl'] or 'not recorded'} mg/dL, BMI {m['bmi'].get('value') or 'not available'}"
         f"{' (' + m['bmi']['who'] + ')' if m['bmi'].get('who') else ''}.", "",
         "DIABETES ASSESSMENT:", f"{d['status']}. Hybrid model score {d['hybrid_score']} (threshold {d['hybrid_threshold']}); "
         f"{d['models_flagging']} models flag elevated risk."]
    if d["criteria_met"]:
        L.append("Criteria met: " + "; ".join(d["criteria_met"]) + ".")
    L += ["", "HYPERTENSION ASSESSMENT:", h["status"] + ".", "", "CURRENT MEDICATIONS:"]
    meds = p["nlp_extraction"].get("medications") or []
    L += [f"- {x.get('text')} {x.get('dose') or ''} {x.get('frequency') or ''}".rstrip() for x in meds] or ["No current medications recorded."]
    L += ["", "RECOMMENDED INVESTIGATIONS:", "Fasting plasma glucose and HbA1c to confirm glycaemic status; lipid profile; serum creatinine.",
          "", "DATA QUALITY AND LIMITATIONS:"]
    L += [f"- {i['field']}: {i['message']}" for i in p["data_issues"]] or ["No data issues."]
    L += ["", "AI-assisted decision support - to be verified by the treating clinician."]
    return "\n".join(L)
