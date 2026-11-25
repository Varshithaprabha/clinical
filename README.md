# Diabetes Screening – Clinician Workspace

This project screens a patient for diabetes, and also hypertension. It combines machine-learning models on the patient's measurements, clinical NLP on the clinician's free-text notes, and an LLM that writes a doctor-style assessment report in plain text.

## Run

```powershell
pip install -r requirements.txt
python -m src.train        # trains every model and writes reports/metrics.md (one-off, ~20-30 min on CPU)
python app.py              # open http://127.0.0.1:5000
```

The LLM needs `GROQ_API_KEY` set in the environment. The default model is `openai/gpt-oss-120b`. To use any other OpenAI-compatible endpoint, set `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, `OPENAI_BASE_URL` and `LLM_MODEL`. Without an LLM, the app still works: it falls back to the rule-based NLP extractor and a template report.
The patient name and ID are never sent to the LLM.

## GUI (as specified in Updates.docx)
1. **Screening.** Patient identification, then fields 01–13: personal history, medications, family history, measurements with automatic BMI, vision, lower limbs, advice and follow-up. Also Save draft and Continue to processing.
   - **Added fields:**
     - Sex. When filled in, it overrides the notes; if the two disagree, a warning is shown.
     - Pregnancy status, shown only for women.
     - Known drug allergies.
     - Optional fasting glucose, HbA1c and eGFR. The document names fasting glucose and HbA1c as attributes, and eGFR is needed for safe metformin and SGLT2 prescribing.
   - **BMI** updates live as weight and height are typed. It shows the WHO and Asian categories, reads a height like 1.62 as metres (162 cm), and explains when it can't be calculated.
2. **AI processing.** Shows the record that was received and the progress of each module: data validation and missing-data handling, BMI, clinical NLP extraction, feature preparation, models, and the report.
3. **Assessment.** Calculated BMI, the hypertension assessment, the diabetes assessment (with the score from every model), and the NLP extraction review, where the clinician can verify the extracted features and re-run the assessment. Also Explainable AI (SHAP + LIME) and the **LLM clinical report in text form**, which can be copied, downloaded as .txt, or **downloaded as a formatted PDF** (patient details, measurements, assessment summary, model scores, SHAP factors, the full report, and the prescription exactly as the clinician edited it, marked DRAFT until approved). A sample is in `reports/sample_report_patient_A.pdf`.

## Models (all mandatory ones are included)
| Model | Data |
|---|---|
| Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM (+ KNN, NB, LDA, GB, AdaBoost, ExtraTrees, MLP, HistGB) | Pima, DiaBD, and the combined "general" dataset |
| TF-IDF + SVM (NLP baseline) | Clinical notes |
| BioBERT embeddings + classifier | Clinical notes (`dmis-lab/biobert-base-cased-v1.1`) |
| Hybrid XGBoost: structured + NLP-extracted + BioBERT features | Primary diabetes model in the app |
| XGBoost hypertension model | DiaBD `hypertensive` |

For a walkthrough of how everything fits together, see `docs/HOW_IT_WORKS.md`. For results, see `reports/metrics.md`. For the dataset analysis, similar attributes, NLP tasks and other datasets, see `reports/dataset_report.md`.

## Missing values
Missing values are never just dropped or filled with a constant. The pipeline:
1. Converts disguised missing values to NaN.
2. Adds missing-indicator columns.
3. Fills gaps with MICE (iterative model-based imputation), fitted on the training data only.
4. Lets XGBoost handle NaN natively; it is trained with masked copies of the data.
5. Abstains when the required inputs are absent. Details are in the dataset report.

## Layout
```
app.py                 Flask server
templates/index.html   GUI
src/preprocess.py      cleaning, harmonisation, MICE
src/nlp_extract.py     rule-based clinical NLP (negation, meds, family, labs)
src/notes.py           note generator used to train the text models
src/text_models.py     BioBERT embeddings
src/llm.py             LLM extraction + clinician report
src/inference.py       end-to-end assessment logic
src/prescription.py    formulary-checked prescription drafting
src/report_pdf.py      PDF export of the report
src/train.py           training + evaluation
```

This tool is decision support only. Every output must be verified by a clinician.
