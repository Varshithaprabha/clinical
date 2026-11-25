"""Synthetic clinician notes for training/evaluating the NLP models.

Neither dataset ships free text, so a note is written for every DiaBD record in the same
four sections as the clinician form (complaints / existing illness / medications / family
history). IMPORTANT - no leakage: the note is generated ONLY from the record's recorded
history attributes (hypertensive, cardiovascular_disease, stroke, family_diabetes,
family_hypertension, gender, age). The diabetes label is never used, and complaints are
sampled independently of the label, so the text models can only learn what a real history
note would legitimately contain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_STATE

COMPLAINTS = [
    "Routine health check-up", "No active complaints", "Occasional headache for 1 week",
    "General weakness", "Knee joint pain", "Low back pain for 2 months", "Mild cough for 3 days",
    "Dizziness on standing", "Follow-up visit", "Acidity and bloating after meals",
    "Burning sensation in chest", "Came for screening camp", "Mild fever since yesterday",
    "Neck pain", "Body ache", "Sleep disturbance", "Itching over skin",
]
HTN_TERMS = ["Known case of HTN", "Hypertension for {y} years", "k/c/o high blood pressure",
             "Hypertensive since {y} years", "HTN on treatment"]
CVD_TERMS = ["IHD", "Coronary artery disease, post PTCA", "History of MI {y} years back",
             "Angina on exertion", "Ischaemic heart disease"]
STROKE_TERMS = ["Old CVA", "History of stroke {y} years ago", "Previous TIA", "Left hemiparesis after stroke"]
NO_ILLNESS = ["No known comorbidities", "NIL", "Nil significant", "No CVD / DM / HTN / CA / thyroid",
              "No previous illness or surgery", "Denies HTN, CVD or thyroid disease"]
SURGERY = ["Appendectomy in childhood", "Cataract surgery", "LSCS", "Hernia repair", ""]
ANTIHTN = ["Tab Amlodipine 5 mg OD", "Tab Telmisartan 40 mg OD", "Tab Losartan 50 mg BD",
           "Tab Atenolol 50 mg OD", "Tab Enalapril 5 mg BD"]
CVD_MEDS = ["Tab Aspirin 75 mg OD", "Tab Atorvastatin 20 mg HS", "Tab Clopidogrel 75 mg OD"]
OTHER_MEDS = ["Tab Pantoprazole 40 mg OD", "Calcium supplement OD", "Paracetamol SOS", "Vitamin D weekly"]
RELATIVES = ["Mother", "Father", "Elder brother", "Sister", "Paternal grandfather", "Maternal uncle"]
FAM_DM = ["{r} has diabetes", "{r} - type 2 DM", "{r} on insulin for diabetes", "Family history of DM ({r})"]
FAM_HTN = ["{r} is hypertensive", "{r} has high BP", "{r} - HTN"]
FAM_NONE = ["No relevant family history", "Not significant", "Denies family history of DM / HTN",
            "Nil", "No history of diabetes or hypertension in family"]


def _pick(rng, xs, **kw):
    return rng.choice(xs).format(**kw)


def note_for_row(row: pd.Series, rng: np.random.Generator) -> dict:
    y = int(rng.integers(1, 15))
    complaints = _pick(rng, COMPLAINTS)

    ill = []
    if row.get("hypertensive") == 1:
        ill.append(_pick(rng, HTN_TERMS, y=y))
    if row.get("cardiovascular_disease") == 1:
        ill.append(_pick(rng, CVD_TERMS, y=y))
    if row.get("stroke") == 1:
        ill.append(_pick(rng, STROKE_TERMS, y=y))
    if not ill:
        ill.append(_pick(rng, NO_ILLNESS))
    s = _pick(rng, SURGERY)
    if s and rng.random() < 0.3:
        ill.append("Surgery: " + s)
    existing = ". ".join(ill)

    meds = []
    if row.get("hypertensive") == 1 and rng.random() < 0.85:
        meds.append(_pick(rng, ANTIHTN))
    if row.get("cardiovascular_disease") == 1:
        meds += list(rng.choice(CVD_MEDS, size=2, replace=False))
    if rng.random() < 0.2:
        meds.append(_pick(rng, OTHER_MEDS))
    medications = ", ".join(meds) if meds else "NIL"

    fam = []
    if row.get("family_diabetes") == 1:
        fam.append(_pick(rng, FAM_DM, r=_pick(rng, RELATIVES)))
    if row.get("family_hypertension") == 1:
        fam.append(_pick(rng, FAM_HTN, r=_pick(rng, RELATIVES)))
    family = ". ".join(fam) if fam else _pick(rng, FAM_NONE)

    return {"complaints": complaints, "existing_illness": existing,
            "medications": medications, "family_history": family}


def note_text(sections: dict) -> str:
    return (f"Complaints: {sections.get('complaints') or 'not recorded'}. "
            f"Existing illness: {sections.get('existing_illness') or 'not recorded'}. "
            f"Medications: {sections.get('medications') or 'not recorded'}. "
            f"Family history: {sections.get('family_history') or 'not recorded'}.")


def build_notes(df: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [note_for_row(r, rng) for _, r in df.iterrows()]
    out = pd.DataFrame(rows, index=df.index)
    out["text"] = out.apply(lambda r: note_text(r.to_dict()), axis=1)
    return out
