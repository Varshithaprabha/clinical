# Sample patients for testing the Clinician Workspace

Type these into the Screening form (fields not listed can stay blank).
The names are made up.

## 1. Ramesh Kumar: diabetes with ADA criteria met
| Field | Value |
|---|---|
| Age / Sex | 58 / Male |
| Weight / Height | 82 kg / 168 cm |
| BP / GRBS | 152/94 / 246 |
| HbA1c | 8.2 |
| Complaints | Excessive thirst and frequent urination for 3 months, tiredness |
| Existing illness | Hypertension for 6 years. Old MI in 2019. No h/o stroke. |
| Medications | Amlodipine 5 mg OD |
| Family history | Father diabetic, mother hypertensive |
| Allergies | NKDA |
| Vision / Lower limbs | Blurred vision occasionally / Tingling in both feet |
| Advice / Follow-up | Diet control / 2 weeks |

Expected: Diabetes **High** (criteria met), Hypertension **Known**, BMI 29.1. The prescription adds an ARB, metformin, a statin and aspirin, with a warning to check kidney function.

## 2. Lakshmi Devi: diabetes with poor kidney function
| Field | Value |
|---|---|
| Age / Sex | 64 / Female, Pregnancy: No |
| Weight / Height | 70 kg / 1.55 (typed in metres, so it is read as 155 cm) |
| BP / GRBS | 138/86 / 212 |
| Fasting glucose / HbA1c / eGFR | 148 / 7.6 / 38 |
| Complaints | Frequent urination at night, burning feet |
| Existing illness | Chronic kidney disease stage 3 |
| Medications | (blank) |
| Family history | Sister diabetic |
| Allergies | Sulfa drugs |

Expected: metformin is **not started** because eGFR is 38. Empagliflozin is chosen instead.

## 3. Priya S: pregnant with severe high BP
| Field | Value |
|---|---|
| Age / Sex | 29 / Female, Pregnancy: **Pregnant** |
| Weight / Height | 68 kg / 160 cm |
| BP / GRBS | 166/112 / 155 |
| Complaints | Headache and swelling of feet for 1 week |
| Existing illness | No known illness |
| Allergies | None known |

Expected: **URGENT** warning for severe hypertension in pregnancy. No drugs are started automatically, and she is referred to obstetric care.

## 4. Arjun Mehta: healthy young adult (low risk)
| Field | Value |
|---|---|
| Age / Sex | 26 / Male |
| Weight / Height | 68 kg / 175 cm |
| BP / GRBS | 118/76 / 98 |
| Complaints | Routine health check, no complaints |
| Existing illness | No known illness. No h/o diabetes or hypertension. |
| Family history | No family history of diabetes |
| Allergies | NKDA |

Expected: Diabetes **Low**, BP Normal, BMI about 22.2, and no medication.

## 5. Fatima Begum: borderline (model-based risk)
| Field | Value |
|---|---|
| Age / Sex | 47 / Female, Pregnancy: No |
| Weight / Height | 84 kg / 158 cm |
| BP / GRBS | 134/88 / 172 |
| Complaints | Tiredness, weight gain |
| Existing illness | Hypothyroidism. Diabetes ruled out last year. |
| Medications | Thyroxine 50 mcg OD |
| Family history | Mother and brother diabetic |
| Allergies | Penicillin |

Expected: no ADA criterion is met, so the models decide (Moderate or High risk). BMI is about 33.6 (Obese). In the NLP review, "Diabetes" is shown struck through because it was "ruled out".

## 6. Unknown patient: missing data
| Field | Value |
|---|---|
| Age / Sex | 52 / (leave blank) |
| Height | 165 cm (no weight) |
| Everything else | blank |

Expected: BMI shows "Enter weight to calculate", and Diabetes shows **Insufficient data**. The system refuses to guess.
