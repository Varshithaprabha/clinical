# Dataset report – diabetes screening project

This report covers the items in `Updates.docx`: dataset details, similar attributes, NLP tasks to add, and other datasets.
The model results are in [metrics.md](metrics.md).

---

## 1. Dataset A – `diabetes.csv` (Pima Indians Diabetes, extended copy)

| Item | Value |
|---|---|
| Origin | US National Institute of Diabetes and Digestive and Kidney Diseases. The cohort is Pima Indian women aged 21 or older. |
| Rows in the file | 9,054 |
| **Exact duplicate rows** | **8,276 (91%)**. Only **778 rows are unique** (the original 768-row dataset plus a few extra rows). |
| Target | `Outcome` (1 = diabetes within 5 years). In the unique rows, 34% are positive. |
| Features | Pregnancies, Glucose (2-h OGTT plasma glucose, mg/dL), BloodPressure (**diastolic** mmHg), SkinThickness (triceps mm), Insulin (2-h serum µU/mL), BMI, DiabetesPedigreeFunction, Age |

**Data-quality findings**
* **Duplicates.** If the duplicates are kept, the same patient lands in both the training and the test split. That inflates accuracy to around 99%, and the number is not real. All analysis here uses the 778 unique rows.
* **Missing values stored as zeros.** A value of 0 is physically impossible for these measures:
  Glucose 64, BloodPressure 416, SkinThickness 2,618, Insulin 4,354, BMI 128 zeros in the raw file (counts after de-duplication are in `metrics.json`). These zeros are converted to *missing*, not treated as real measurements.
* The cohort is women only, from one ethnic group. The glucose value is a 2-hour OGTT result, not a random glucose.

## 2. Dataset B – DiaBD (Bangladesh)

| Item | Value |
|---|---|
| Origin | DiaBD: a diabetes dataset for risk analysis and research in Bangladesh (hospital and community screening). |
| Rows | 5,288, with no duplicates |
| Target | `diabetic` (Yes/No). Only **342 positives (6.5%)**, so the classes are heavily imbalanced. |
| Features | age, gender, pulse_rate, systolic_bp, diastolic_bp, glucose (**mmol/L**), height (m), weight (kg), bmi, family_diabetes, hypertensive, family_hypertension, cardiovascular_disease, stroke |

**Data-quality findings**
* No values are recorded as NaN, but some values are impossible:
  height 0.36 m, weight 3 kg, BMI 574, pulse 5/min, and glucose of 0–1.4 mmol/L.
  Values outside clinical plausibility ranges are set to *missing*. BMI is **recomputed** from validated height and weight.
* Glucose is converted to mg/dL (×18.016) so it is in the same unit as the GUI's GRBS field and Dataset A.
* This dataset matches the clinician form closely: age, height, weight, BP, blood glucose, and history attributes that also appear in the free-text fields.

## 3. Similar attributes and how they were harmonised

| Concept | Pima | DiaBD | Clinician GUI | Harmonised feature |
|---|---|---|---|---|
| Age | Age | age | Age (field 4) | `age` |
| BMI | BMI | bmi (recomputed) | computed from fields 5 and 6 | `bmi` |
| Blood glucose | Glucose (mg/dL, OGTT) | glucose (mmol/L → mg/dL) | GRBS (mg/dL) | `glucose` |
| Diastolic BP | BloodPressure | diastolic_bp | BP field (x/**y**) | `diastolic_bp` |
| Systolic BP | – | systolic_bp | BP field (**x**/y) | `systolic_bp` |
| Sex | all female | gender | from text (NLP) | `gender_male` |
| Family history of diabetes | DiabetesPedigreeFunction (a genetic score; related but not the same) | family_diabetes | Family history text → NLP | `family_diabetes` |
| Hypertension history | – | hypertensive | Existing illness / medication text → NLP | `hypertensive` |
| CVD, stroke, family HTN | – | yes | Existing illness / family text → NLP | same names |
| Pregnancies, Insulin, SkinThickness | yes | – | – | Pima-specific model only |
| HbA1c, fasting glucose | – | – | Free text → NLP (lab regex or LLM) | ADA criteria layer |

An attribute that a source never recorded is left **structurally missing** in the combined data. Missing-aware models handle it. It is never filled with a constant.

## 4. Missing-value handling (no plain dropping or constant filling)

1. **Disguised missing values become NaN:** zeros in Pima, and out-of-range values in DiaBD.
2. **Missing-indicator columns** are added, because the fact that a test was not done can carry information.
3. **MICE (IterativeImputer with BayesianRidge)** estimates each missing value from the patient's *other* attributes. It is fitted on training folds only, inside the pipeline, so nothing leaks from the test data. Used for LR, DT, RF, SVM and the other sklearn models.
4. **XGBoost handles NaN natively.** It learns which branch a missing value should follow. Training also includes a copy of the data with 20% of cells masked at random, so the model behaves sensibly when a clinician leaves a field empty.
5. **The app can abstain.** Without a glucose value (GRBS, FPG or HbA1c) the diabetes assessment is withheld. Without a BP reading the hypertension assessment is withheld. In both cases the report says *what* is missing.
6. Evidence for this approach: `metrics.md` compares complete-case dropping, median filling, KNN, MICE and XGBoost-native handling. It also shows how ROC-AUC degrades as 10–50% of test cells are blanked.

## 5. NLP tasks added

| Task | Where it is used |
|---|---|
| Clinical named-entity recognition: diseases (CVD / DM / HTN / CA / thyroid, CKD, dyslipidaemia), symptoms, medications | Personal history, medication and family-history fields |
| **Negation detection** ("No DM", "denies HTN", "hypothyroidism ruled out") | Rule-based NegEx-style detector, plus the LLM |
| Abbreviation normalisation (k/c/o, HTN, DM, IHD, CVA, OD/BD/HS) | Lexicon and LLM |
| Medication parsing: name, dose, frequency and drug class (antidiabetic, antihypertensive, statin, antiplatelet, thyroid) | Medication field |
| Relation extraction for family history (relative ↔ condition) | Family-history field |
| Lab-value extraction (HbA1c %, FBS/FPG, PPBS, with mmol/L → mg/dL conversion) | Any text field. Feeds the ADA criteria check |
| Note classification: **TF-IDF + SVM** (baseline) and **BioBERT embeddings + classifier** | Whole note |
| **Hybrid XGBoost**: structured measurements + NLP-extracted flags + BioBERT embedding (16 PCA components) | Primary diabetes model |
| **LLM tasks** (Groq `openai/gpt-oss-120b`): structured JSON extraction, and writing the clinician-style assessment text | NLP review card and the clinical report |

Neither dataset includes free text. For training, a clinical note was generated for every DiaBD record (`src/notes.py`). Each note is built **only from that record's history attributes**. The diabetes label is never used, and complaints are sampled independently of the label, so the text cannot leak the answer. A sample is in `synthetic_clinical_notes_sample.csv`. Because the notes are generated, the text-model scores show that the pipeline works end to end. They are not evidence of accuracy on real clinical notes, which would need real notes to validate.

## 6. Other datasets worth adding

| Dataset | What it adds |
|---|---|
| **CDC BRFSS Diabetes Health Indicators** (Kaggle, about 250k survey rows) | Large general population. Includes BP, cholesterol, BMI, lifestyle and age group. |
| **NHANES** (CDC) | Lab-measured **HbA1c, fasting glucose** and OGTT, plus BP, BMI and medications. The best fit for the HbA1c/FPG attributes in the brief. |
| **Early-stage diabetes risk prediction** (UCI, Sylhet, Bangladesh, 520 rows) | Symptoms such as polyuria, polydipsia, weight loss and blurred vision. Useful for the symptom/NLP features. |
| **Diabetes 130-US hospitals** (UCI, 100k encounters) | Diagnoses, medications (insulin, metformin…) and HbA1c results. Medication-NLP signal. |
| **MIMIC-IV / MIMIC-IV-Note** (PhysioNet, credentialed access) | Real discharge summaries. The right corpus to validate clinical NLP and BioBERT on real text. |
| **n2c2 / i2b2 2014 Heart Disease Risk Factors** (credentialed) | Annotated notes for diabetes, HTN, CAD, medications and family history. A gold standard for the extractor. |
| **Mendeley "Diabetes Dataset" (Iraq, 1,000 rows)** | HbA1c, lipids, urea and creatinine. |
