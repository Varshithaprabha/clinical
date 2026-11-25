# How this project works

This is a plain-language walkthrough of the diabetes screening project: what each part does, why it was built that way, and how to run and demo it. Read it before presenting the project.

---

## 1. The big picture

A clinician fills in a screening form. The system then:

1. checks the data;
2. reads the free-text notes with NLP;
3. runs machine-learning models;
4. applies clinical guideline rules;
5. drafts a prescription;
6. asks an LLM to write a doctor-style report.

The clinician reviews everything and approves it.

```
 Screening form (Step 1)
   │  age, sex, weight, height → BMI (live), BP, GRBS, optional FBS/HbA1c/eGFR,
   │  free text: complaints, existing illness, medications, family history, allergies
   ▼
 /api/assess  (src/inference.py)
   ├─ 1. Validate          ranges, unit conversion (m→cm, mmol/L→mg/dL), missing fields flagged
   ├─ 2. Clinical NLP      LLM JSON extraction  (fallback: rule-based NegEx extractor)
   │                       → conditions, symptoms, meds, family history, labs, negations
   ├─ 3. Models            Hybrid XGBoost (structured + NLP + BioBERT)   ← primary
   │                       LR, DT, RF, XGBoost, SVM (structured), TF-IDF+SVM, BioBERT+LR (text)
   │                       Hypertension XGBoost
   ├─ 4. Guideline layer   ADA diabetes criteria, ACC/AHA BP categories → final status/level
   ├─ 5. Explainability    SHAP + LIME on the hybrid model
   ├─ 6. Prescription      LLM proposes → formulary validator → guideline floor
   └─ 7. Report            LLM writes the doctor-style text (fallback: template)
   ▼
 Processing screen (Step 2) → Assessment screen (Step 3)
   verify NLP features → re-assess · approve prescription · copy / .txt / PDF
```

---

## 2. The data (`src/preprocess.py`)

| Dataset | What it has | Problems found and fixed |
|---|---|---|
| `diabetes.csv` (Pima-style) | 8 measurements and an outcome | 8,276 of 9,054 rows were exact duplicates, so 778 unique rows remain. A value of 0 for glucose, BP, skin, insulin or BMI really means "missing". |
| DiaBD (Bangladesh) | Age, sex, BP, glucose, BMI, history (HTN, CVD, stroke, family) and diabetes | Glucose is recorded in mmol/L and converted to mg/dL (×18.016). Implausible values are set to missing, and BMI is recomputed from height and weight. |

**"General" model.** The attributes both datasets share (age, BMI, BP, glucose and so on) are harmonised into one feature set. A model trained on both datasets generalises better than one trained on either alone. This answers the brief that the system should "work in general".

### Missing values (a key requirement)

The project never just fills a blank with the mean or drops the row:

1. **Disguised missing values become NaN.** For example, a BMI of 0 is impossible, so it counts as missing.
2. **Missing-indicator columns.** Whether a value is missing can itself be informative, so the model sees it.
3. **MICE imputation.** Each missing value is predicted from the other features (IterativeImputer with BayesianRidge). It is fitted on training data only, so there is no leakage.
4. **XGBoost handles NaN natively.** During training, 20% of cells are randomly blanked, so the model learns to cope with incomplete patients.
5. **Abstention.** If too much is missing (for example, no glucose and no history), the system says "Insufficient data" instead of guessing.

Evidence for these choices is in `reports/metrics.md`. Complete-case analysis would keep only 397 of 778 Pima rows. MICE scores about the same AUC as median or KNN imputation. Random Forest with MICE still reaches AUC 0.83 with 50% of cells missing.

---

## 3. The text side (NLP)

The datasets have no clinical notes. To train text models, **synthetic notes** are generated from each DiaBD patient's history fields (`src/notes.py`). Example: *"Known case of hypertension. No history of stroke. Mother diabetic."*

The notes are built from history only, never from the diabetes label, so the label cannot leak into the text.

| Component | File | What it does |
|---|---|---|
| Rule-based extractor | `src/nlp_extract.py` | Finds conditions, symptoms, medications (with dose and frequency), family history and lab values (HbA1c, FBS) using lexicons and regex. It handles **negation**: "no h/o stroke" and "CAD ruled out" are recognised as negative. |
| LLM extractor | `src/llm.py` | Groq `openai/gpt-oss-120b` returns the same JSON schema. The rule-based extractor is used if the LLM is unavailable. |
| TF-IDF + SVM | `src/train.py` | Classic text baseline. |
| BioBERT + classifier | `src/text_models.py` | Turns each note into a 768-dimensional embedding (`dmis-lab/biobert-base-cased-v1.1`), followed by Logistic Regression. |
| Hybrid XGBoost | `src/train.py` | Combines structured values, NLP-extracted flags (hypertensive, CVD, family history, symptoms…) and 16 BioBERT PCA components. This is the main model in the app. |

---

## 4. The models and results (`src/train.py`, `reports/metrics.md`)

All mandatory models are included, plus extras (KNN, Naive Bayes, LDA, GB, AdaBoost, ExtraTrees, MLP, HistGB). The evaluation setup:

- a stratified 80/20 split;
- 5-fold cross-validation on the training split;
- decision thresholds tuned on out-of-fold predictions only.

| Model | Test ROC-AUC (approx.) |
|---|---|
| LR / DT / RF / XGBoost / SVM on the general dataset | 0.86 – 0.88 |
| Same models on DiaBD only | 0.83 – 0.86 |
| Same models on Pima only | 0.73 – 0.78 |
| TF-IDF + SVM (text only) | 0.73 |
| BioBERT + LR (text only) | 0.71 |
| Hybrid XGBoost | 0.82 (PR-AUC 0.34, better than 0.26 for measurements only) |
| Hypertension XGBoost | 0.86 |

**Why the hybrid model is primary:** its ROC-AUC is similar, but its PR-AUC is clearly better. PR-AUC matters more because diabetic patients are the minority class. The hybrid model also uses the clinician's notes, which a real doctor would use.

---

## 5. From model score to a clinical decision (`src/inference.py`)

ML probabilities alone are not a diagnosis, so a guideline layer sits on top. The checks run in this order:

1. **Known diabetes** (in the history or on diabetes medication) → "Known condition".
2. **ADA criteria met**: HbA1c ≥ 6.5%, FPG ≥ 126 mg/dL, or random glucose ≥ 200 mg/dL with classic symptoms → "High, criteria met, confirm with repeat test".
3. **Insufficient data** → abstain.
4. **Otherwise use the models.** "High" needs the hybrid score above its tuned threshold **and** a majority of the models agreeing. Otherwise the result is "Moderate" or "Low".

Hypertension works the same way: known history, then ACC/AHA BP category, then the model.

**Sex:** the form value is authoritative. If the notes say otherwise, a warning is shown.

---

## 6. Explainability

- **SHAP** (TreeExplainer on the hybrid model) shows how much each feature pushed this patient's risk up or down. The 16 BioBERT components are grouped into one row, "Clinical text (BioBERT context)".
- **LIME** gives human-readable rules, such as "GRBS > 144.85 raises risk".

---

## 7. The prescription, written like a doctor (`src/prescription.py`)

An LLM must never be trusted to prescribe on its own, so prescribing is built in three safety layers:

1. **The LLM proposes** drugs, doses and frequencies, for example `1-0-1` or `0-0-1`.
2. **The formulary validator** checks every item against a fixed list of 11 drugs. It checks:
   - allowed dose and frequency, and the maximum daily dose;
   - eligibility: no diabetes drug without diabetes, no BP drug without hypertension, statins only where indicated;
   - contraindications: allergies (including the drug class), pregnancy, kidney function (metformin is blocked below eGFR 30 and not started below 45), age ≥ 65 for glimepiride, and aspirin only with CVD or stroke.

   Anything unsafe is **removed and listed** with the reason.
3. **The guideline floor.** If a confirmed condition is left untreated (for example, the LLM forgot metformin), the rule engine adds the standard choice, labelled "added by guideline engine".

Warnings are also raised for:
- eGFR not recorded;
- a single high BP reading;
- severe hypertension in pregnancy;
- allergy status unknown;
- GRBS ≥ 300.

**The clinician has the final say.** They can remove items, must tick "I have reviewed…" and enter their name before "Approve & print". Until then, the PDF marks the prescription as DRAFT.

---

## 8. The LLM report and PDF

- `src/llm.py` sends the structured findings (never the patient name or ID) to the LLM. It asks for fixed sections: Clinical summary, Diabetes, Hypertension, BMI, Key factors, Medication review, Investigations, Lifestyle/follow-up, and Data quality.
- If no LLM is available, a template report is produced instead, so the app never breaks.
- **Download PDF** (`src/report_pdf.py`, route `POST /api/report.pdf`) builds an A4 report. It contains:
  - patient details and measurements;
  - a colour-coded summary;
  - the score of every model and the top SHAP factors;
  - the full clinician report;
  - the prescription as currently edited, with warnings, advice, investigations and a signature line.

---

## 9. How to run and demo

```powershell
cd C:\Projects\Intern_Project\Intern_Project
pip install -r requirements.txt
$env:GROQ_API_KEY = "<your key>"     # optional – without it the rule-based NLP + template report are used
python app.py                        # then open http://127.0.0.1:5000
```

Re-training is only needed if the data or code changes. It takes about 20–30 minutes on CPU, and `--fresh` ignores the checkpoint:

```powershell
python -m src.train
```

**Suggested demo patients**

| Patient | Inputs | What it shows |
|---|---|---|
| A: diagnosed diabetic | 58 M, 82 kg / 168 cm, BP 152/94, GRBS 246, HbA1c 8.2; thirst + frequent urination; HTN 6 yrs, old MI; Amlodipine 5 mg OD; father diabetic | Criteria met, full prescription with empagliflozin (because of CVD), statin and aspirin, SHAP led by GRBS |
| B: kidney | Same as A, plus eGFR 38 | Metformin not started; empagliflozin chosen instead |
| C: pregnancy | 30 F, pregnant, BP 165/112 | No automatic drugs, URGENT warning, referral |
| D: missing data | Only age and height | "Insufficient data": the system abstains instead of guessing |
| E: negation | Notes say "no h/o stroke, CAD ruled out" | NLP review shows them struck through, not counted |

Also show the following:
- live BMI (type 1.68 as the height and it is read as 168 cm);
- the NLP review, where you change a feature and re-assess;
- removing a prescription item, approving it, and downloading the PDF.

---

## 10. Questions you may be asked

- **Why not just drop rows with missing values?** Pima would lose about half its rows (397/778 left), and real patients arrive with gaps. MICE plus missing indicators and native NaN handling keep the information. Robustness tests show the models degrade gracefully.
- **Why synthetic notes?** Neither dataset has free text. The notes are generated only from history fields, never the label, so the text models learn from clinical language, not from the answer.
- **Why does BioBERT score lower than XGBoost?** The notes are short and templated, so a large language model has little extra to learn from them. Its value is as context inside the hybrid model; the ablation in `metrics.md` shows this.
- **Can the LLM prescribe something dangerous?** No. Every item passes the formulary validator, unsafe items are removed with the reason shown, and nothing is final until a clinician approves it.
- **Is patient data sent to the cloud?** Only de-identified clinical values go to the LLM. The patient name and ID are never sent.
- **Limitations:**
  - The datasets are small and from specific populations (Pima Indian women; Bangladesh).
  - The notes are synthetic.
  - The formulary is limited to 11 drugs.
  - Prospective validation on real clinic data would be needed before any real use.

---

## 11. File map

```
app.py                     Flask server: /, /api/health, /api/assess, /api/report.pdf
templates/index.html       3-step GUI (Screening → AI processing → Assessment)
src/config.py              paths, model names, LLM settings
src/preprocess.py          cleaning, harmonisation, MICE (SafeIterativeImputer)
src/notes.py               synthetic clinical-note generator
src/nlp_extract.py         rule-based clinical NLP with negation
src/text_models.py         BioBERT embeddings
src/train.py               training, evaluation, robustness tests → models/, reports/
src/inference.py           end-to-end assessment for one patient
src/prescription.py        formulary, safety rules, guideline floor
src/llm.py                 LLM extraction and report writing
src/report_pdf.py          PDF export
reports/                   metrics.md, dataset_report.md, sample reports (.txt and .pdf)
models/                    trained models (.joblib) and BioBERT cache
```
