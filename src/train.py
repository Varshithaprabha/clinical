"""Train and evaluate every model required by the project brief.

Run:  python -m src.train
Outputs: models/*.joblib, reports/metrics.json, reports/metrics.md
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import nlp_extract, notes, text_models
from .config import (BIOBERT_PCA_COMPONENTS, CACHE_DIR, CV_FOLDS, MODELS_DIR, RANDOM_STATE,
                     REPORTS_DIR, TEST_SIZE)
from .preprocess import (DIABD_FEATURES, GENERAL_FEATURES, HTN_FEATURES, PIMA_FEATURES, load_combined,
                         load_diabd, load_pima, make_mice_imputer)

warnings.filterwarnings("ignore")
RS = RANDOM_STATE
MANDATORY = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "Support Vector Machine"]

HYBRID_STRUCT = ["age", "gender_male", "bmi", "glucose", "systolic_bp", "diastolic_bp"]
HYBRID_NLP = ["hypertensive", "cardiovascular_disease", "stroke", "family_diabetes",
              "family_hypertension", "symptom_count"]


# ---------------------------------------------------------------- helpers
def mice_pipe(clf):
    return Pipeline([("impute", make_mice_imputer()), ("scale", StandardScaler()), ("clf", clf)])


def xgb(pos_weight: float = 1.0):
    return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
                         colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0,
                         scale_pos_weight=pos_weight, eval_metric="logloss",
                         random_state=RS, n_jobs=-1)  # NaN handled natively


def classifiers(pos_weight: float) -> dict:
    """Mandatory five + every other common classifier family ("all possible classifiers")."""
    return {
        "Logistic Regression": mice_pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Decision Tree": mice_pipe(DecisionTreeClassifier(max_depth=5, min_samples_leaf=10,
                                                          class_weight="balanced", random_state=RS)),
        "Random Forest": mice_pipe(RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                                          class_weight="balanced_subsample",
                                                          random_state=RS, n_jobs=-1)),
        "XGBoost": xgb(pos_weight),
        "Support Vector Machine": mice_pipe(SVC(kernel="rbf", C=1.0, probability=True,
                                                class_weight="balanced", random_state=RS)),
        "K-Nearest Neighbours": mice_pipe(KNeighborsClassifier(n_neighbors=15, weights="distance")),
        "Gaussian Naive Bayes": mice_pipe(GaussianNB()),
        "LDA": mice_pipe(LinearDiscriminantAnalysis()),
        "Gradient Boosting": mice_pipe(GradientBoostingClassifier(random_state=RS)),
        "AdaBoost": mice_pipe(AdaBoostClassifier(n_estimators=200, random_state=RS)),
        "Extra Trees": mice_pipe(ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3,
                                                      class_weight="balanced", random_state=RS, n_jobs=-1)),
        "MLP Neural Net": mice_pipe(MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1500,
                                                  early_stopping=True, random_state=RS)),
        "HistGradientBoosting (native NaN)": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RS),
    }


def best_threshold(y, p) -> float:
    """Threshold maximising F1 (chosen on training out-of-fold predictions)."""
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[np.argmax([f1_score(y, p >= t, zero_division=0) for t in grid])])


def metrics(y, p, thr: float = 0.5) -> dict:
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"threshold": round(thr, 3),
            "accuracy": accuracy_score(y, yhat), "balanced_accuracy": balanced_accuracy_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0), "recall_sensitivity": recall_score(y, yhat),
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0, "f1": f1_score(y, yhat),
            "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def oof_proba(model, X, y, folds=CV_FOLDS):
    skf = StratifiedKFold(folds, shuffle=True, random_state=RS)
    p = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = clone(model).fit(X.iloc[tr] if hasattr(X, "iloc") else X[tr], y[tr])
        p[va] = m.predict_proba(X.iloc[va] if hasattr(X, "iloc") else X[va])[:, 1]
    return p


def mask_augment(X: pd.DataFrame, y: np.ndarray, cols: list[str], rate=0.2, seed=RS):
    """Add a copy of the training rows with random cells blanked so the model learns how to
    behave when the clinician leaves a field empty (missingness robustness)."""
    rng = np.random.default_rng(seed)
    Xm = X.copy()
    for c in cols:
        Xm.loc[rng.random(len(Xm)) < rate, c] = np.nan
    return pd.concat([X, Xm], ignore_index=True), np.concatenate([y, y])


def r(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


# ---------------------------------------------------------------- experiments
def run_structured(name, X, y, results, save=False):
    print(f"\n=== {name}: {X.shape}, positives={int(y.sum())} ===")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RS)
    pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RS)
    out = {}
    fitted = {}
    for mname, model in classifiers(pw).items():
        t = time.time()
        cv = cross_val_score(model, Xtr, ytr, cv=skf, scoring="roc_auc", n_jobs=1)
        thr = best_threshold(ytr, oof_proba(model, Xtr, ytr))
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        out[mname] = {"cv_roc_auc_mean": float(cv.mean()), "cv_roc_auc_std": float(cv.std()),
                      "test_at_0.5": r(metrics(yte, p, 0.5)), "test_at_tuned": r(metrics(yte, p, thr)),
                      "mandatory": mname in MANDATORY}
        fitted[mname] = (model, thr)
        print(f"  {mname:36s} CV AUC {cv.mean():.3f}±{cv.std():.3f} | test AUC {out[mname]['test_at_0.5']['roc_auc']:.3f}"
              f" F1@tuned {out[mname]['test_at_tuned']['f1']:.3f} ({time.time()-t:.1f}s)")
    results[name] = {"n_train": int(len(ytr)), "n_test": int(len(yte)), "features": list(X.columns),
                     "models": out}
    return fitted, (Xtr, Xte, ytr, yte)


def missing_strategy_comparison(results):
    """Why MICE / native handling instead of dropping or constant filling (Pima has most gaps)."""
    p, _ = load_pima()
    X, y = p[PIMA_FEATURES], p["Outcome"].values
    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RS)
    strategies = {
        "Complete-case (drop rows)": None,
        "Median fill": SimpleImputer(strategy="median"),
        "KNN imputation": KNNImputer(n_neighbors=7),
        "MICE + missing indicators": make_mice_imputer(),
    }
    out = {}
    for sname, imp in strategies.items():
        for mname, clf in {"Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
                           "Random Forest": RandomForestClassifier(400, min_samples_leaf=3, random_state=RS, n_jobs=-1)}.items():
            if imp is None:
                keep = X.notna().all(axis=1)
                s = cross_val_score(Pipeline([("s", StandardScaler()), ("c", clf)]), X[keep], y[keep], cv=skf, scoring="roc_auc")
                note = f"only {int(keep.sum())}/{len(X)} rows usable"
            else:
                s = cross_val_score(Pipeline([("i", imp), ("s", StandardScaler()), ("c", clf)]), X, y, cv=skf, scoring="roc_auc")
                note = f"all {len(X)} rows"
            out[f"{sname} | {mname}"] = {"cv_roc_auc": round(float(s.mean()), 4), "std": round(float(s.std()), 4), "rows": note}
    s = cross_val_score(xgb(1.8), X, y, cv=skf, scoring="roc_auc")
    out["XGBoost native missing handling | XGBoost"] = {"cv_roc_auc": round(float(s.mean()), 4), "std": round(float(s.std()), 4), "rows": f"all {len(X)} rows"}
    results["missing_value_strategy_comparison_pima"] = out
    print("\n=== Missing-value strategies (Pima, CV ROC-AUC) ===")
    for k, v in out.items():
        print(f"  {k:60s} {v['cv_roc_auc']:.3f}  ({v['rows']})")


def missingness_robustness(fitted, split, cols, results, key):
    """Blank a growing share of measurement cells in the TEST set and re-score."""
    _, Xte, _, yte = split
    rng = np.random.default_rng(7)
    out = {}
    for rate in [0.0, 0.1, 0.2, 0.3, 0.5]:
        Xm = Xte.copy()
        for c in cols:
            Xm.loc[rng.random(len(Xm)) < rate, c] = np.nan
        out[f"{int(rate*100)}% cells missing"] = {m: round(float(roc_auc_score(yte, fitted[m][0].predict_proba(Xm)[:, 1])), 4)
                                                  for m in MANDATORY}
    results[key] = out
    print(f"\n=== Robustness to missing measurements ({key}) ===")
    for k, v in out.items():
        print(f"  {k:22s} " + "  ".join(f"{m.split()[0]}={a:.3f}" for m, a in v.items()))


def run_nlp_and_hybrid(results):
    d, _ = load_diabd()
    y = d["diabetic"].values
    nts = notes.build_notes(d)
    idx_tr, idx_te = train_test_split(np.arange(len(d)), test_size=TEST_SIZE, stratify=y, random_state=RS)
    ytr, yte = y[idx_tr], y[idx_te]
    nts.to_csv(REPORTS_DIR / "synthetic_clinical_notes_sample.csv", index=False, encoding="utf-8")

    # --- NLP extraction quality (rule-based extractor vs the recorded attributes)
    ext = pd.DataFrame([nlp_extract.extract(**r[["complaints", "existing_illness", "medications", "family_history"]].to_dict())["features"]
                        for _, r in nts.iterrows()], index=d.index)
    results["nlp_extraction_accuracy"] = {c: {"accuracy": round(float(accuracy_score(d[c], ext[c])), 4),
                                              "f1": round(float(f1_score(d[c], ext[c])), 4)}
                                          for c in ["hypertensive", "cardiovascular_disease", "stroke",
                                                    "family_diabetes", "family_hypertension"]}
    texts = nts["text"].tolist()
    out = {}

    # --- 1. TF-IDF + SVM (NLP baseline)
    tfidf_svm = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
                          ("svm", CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=5))])
    Ttr = np.array(texts, dtype=object)[idx_tr]
    Tte = np.array(texts, dtype=object)[idx_te]
    tfidf_svm.fit(Ttr, ytr)
    p = tfidf_svm.predict_proba(Tte)[:, 1]
    out["TF-IDF + SVM (text only)"] = r(metrics(yte, p, best_threshold(ytr, tfidf_svm.predict_proba(Ttr)[:, 1])))
    joblib.dump(tfidf_svm, MODELS_DIR / "tfidf_svm.joblib")

    # --- 2. BioBERT embeddings + classifier
    cache = CACHE_DIR / "biobert_diabd.npy"
    if cache.exists():
        E = np.load(cache)
    else:
        print("  embedding notes with BioBERT (CPU, one-off)...")
        t = time.time()
        E = text_models.embed(texts)
        np.save(cache, E)
        print(f"  done in {time.time()-t:.0f}s")
    bio_clf = Pipeline([("scale", StandardScaler()),
                        ("lr", LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced"))])
    bio_clf.fit(E[idx_tr], ytr)
    p = bio_clf.predict_proba(E[idx_te])[:, 1]
    thr = best_threshold(ytr, oof_proba(bio_clf, E[idx_tr], ytr))
    out["BioBERT embeddings + Logistic Regression (text only)"] = r(metrics(yte, p, thr))
    joblib.dump(bio_clf, MODELS_DIR / "biobert_clf.joblib")

    # --- 3. Hybrid XGBoost: structured measurements + NLP-extracted features + BioBERT PCA
    pca = PCA(n_components=BIOBERT_PCA_COMPONENTS, random_state=RS).fit(E[idx_tr])
    pcs = pd.DataFrame(pca.transform(E), columns=[f"biobert_pc{i+1}" for i in range(BIOBERT_PCA_COMPONENTS)], index=d.index)
    H = pd.concat([d[HYBRID_STRUCT].reset_index(drop=True), ext[HYBRID_NLP].reset_index(drop=True),
                   pcs.reset_index(drop=True)], axis=1)
    hyb_cols = list(H.columns)
    Htr, Hte = H.iloc[idx_tr].reset_index(drop=True), H.iloc[idx_te].reset_index(drop=True)
    pw = float((ytr == 0).sum() / (ytr == 1).sum())
    hybrid = xgb(pw)
    thr_h = best_threshold(ytr, oof_proba(hybrid, Htr, ytr))
    Haug, yaug = mask_augment(Htr, ytr, HYBRID_STRUCT + HYBRID_NLP, rate=0.2)
    hybrid.fit(Haug, yaug)
    p_h = hybrid.predict_proba(Hte)[:, 1]
    out["Hybrid XGBoost (structured + NLP-extracted + BioBERT)"] = r(metrics(yte, p_h, thr_h))

    # ablation: same XGBoost with measurements only (shows what the text adds)
    so = xgb(pw)
    thr_s = best_threshold(ytr, oof_proba(so, Htr[HYBRID_STRUCT], ytr))
    so.fit(Htr[HYBRID_STRUCT], ytr)
    out["Ablation: XGBoost measurements only"] = r(metrics(yte, so.predict_proba(Hte[HYBRID_STRUCT])[:, 1], thr_s))

    results["nlp_and_hybrid_diabd"] = out
    print("\n=== NLP / Hybrid (DiaBD + synthetic notes) ===")
    for k, v in out.items():
        print(f"  {k:58s} AUC {v['roc_auc']:.3f}  PR-AUC {v['pr_auc']:.3f}  F1 {v['f1']:.3f}")

    joblib.dump({"model": hybrid, "pca": pca, "columns": hyb_cols, "threshold": thr_h,
                 "struct": HYBRID_STRUCT, "nlp": HYBRID_NLP,
                 # background sample for SHAP/LIME (rows with NaN filled by column medians only
                 # for the explainer's perturbation distribution - never for prediction)
                 "background": Htr.sample(min(400, len(Htr)), random_state=RS).reset_index(drop=True)},
                MODELS_DIR / "hybrid_xgb.joblib")


def run_hypertension(results):
    d, _ = load_diabd()
    X, y = d[HTN_FEATURES], d["hypertensive"].values.astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RS)
    pw = float((ytr == 0).sum() / (ytr == 1).sum())
    out, saved = {}, {}
    for mname, model in {"XGBoost": xgb(pw),
                         "Logistic Regression": mice_pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
                         "Random Forest": mice_pipe(RandomForestClassifier(400, min_samples_leaf=3, class_weight="balanced_subsample", random_state=RS, n_jobs=-1))}.items():
        thr = best_threshold(ytr, oof_proba(model, Xtr, ytr))
        if mname == "XGBoost":
            Xa, ya = mask_augment(Xtr.reset_index(drop=True), ytr, HTN_FEATURES, 0.2)
            model.fit(Xa, ya)
        else:
            model.fit(Xtr, ytr)
        out[mname] = r(metrics(yte, model.predict_proba(Xte)[:, 1], thr))
        saved[mname] = (model, thr)
    results["hypertension_diabd"] = out
    print("\n=== Hypertension (DiaBD) ===")
    for k, v in out.items():
        print(f"  {k:24s} AUC {v['roc_auc']:.3f}  F1 {v['f1']:.3f}")
    joblib.dump({"model": saved["XGBoost"][0], "threshold": saved["XGBoost"][1], "features": HTN_FEATURES},
                MODELS_DIR / "hypertension_xgb.joblib")


def write_markdown(results, prep):
    L = ["# Model evaluation report", "",
         "All numbers are on a held-out, stratified 20% test split that the models never saw "
         "(5-fold CV ROC-AUC on the training split also shown). Thresholds marked *tuned* were "
         "chosen on out-of-fold training predictions only.", ""]
    L += ["## Data preparation", "", "```json", json.dumps(prep, indent=2), "```", ""]
    for key in ["pima", "diabd", "general"]:
        if key not in results:
            continue
        res = results[key]
        L += [f"## Structured models - {key} (train {res['n_train']}, test {res['n_test']})", "",
              "| Model | CV ROC-AUC | Test ROC-AUC | PR-AUC | Accuracy | Bal. acc | Precision | Recall | F1 (tuned thr) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for m, v in sorted(res["models"].items(), key=lambda kv: -kv[1]["test_at_0.5"]["roc_auc"]):
            t, tt = v["test_at_0.5"], v["test_at_tuned"]
            star = " **(mandatory)**" if v["mandatory"] else ""
            L.append(f"| {m}{star} | {v['cv_roc_auc_mean']:.3f} ± {v['cv_roc_auc_std']:.3f} | {t['roc_auc']:.3f} | {t['pr_auc']:.3f} | "
                     f"{tt['accuracy']:.3f} | {tt['balanced_accuracy']:.3f} | {tt['precision']:.3f} | {tt['recall_sensitivity']:.3f} | {tt['f1']:.3f} |")
        L.append("")
    if "missing_value_strategy_comparison_pima" in results:
        L += ["## Missing-value strategy comparison (Pima, 5-fold CV ROC-AUC)", "", "| Strategy | ROC-AUC | Rows |", "|---|---|---|"]
        for k, v in results["missing_value_strategy_comparison_pima"].items():
            L.append(f"| {k} | {v['cv_roc_auc']:.3f} ± {v['std']:.3f} | {v['rows']} |")
        L.append("")
    if "robustness_general" in results:
        L += ["## Robustness: test ROC-AUC when measurement cells are blanked at random (general model)", "",
              "| Missing | " + " | ".join(MANDATORY) + " |", "|---" * (len(MANDATORY) + 1) + "|"]
        for k, v in results["robustness_general"].items():
            L.append(f"| {k} | " + " | ".join(f"{v[m]:.3f}" for m in MANDATORY) + " |")
        L.append("")
    for key, title in [("nlp_and_hybrid_diabd", "NLP baseline, BioBERT and Hybrid XGBoost (DiaBD + clinical notes)"),
                       ("hypertension_diabd", "Hypertension assessment model (DiaBD)")]:
        if key in results:
            L += [f"## {title}", "", "| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | Specificity | F1 |", "|---|---|---|---|---|---|---|---|"]
            for k, v in results[key].items():
                L.append(f"| {k} | {v['roc_auc']:.3f} | {v['pr_auc']:.3f} | {v['accuracy']:.3f} | {v['precision']:.3f} | "
                         f"{v['recall_sensitivity']:.3f} | {v['specificity']:.3f} | {v['f1']:.3f} |")
            L.append("")
    if "nlp_extraction_accuracy" in results:
        L += ["## Clinical NLP extraction accuracy (rule-based extractor vs recorded attributes)", "",
              "| Attribute | Accuracy | F1 |", "|---|---|---|"]
        for k, v in results["nlp_extraction_accuracy"].items():
            L.append(f"| {k} | {v['accuracy']:.3f} | {v['f1']:.3f} |")
    (REPORTS_DIR / "metrics.md").write_text("\n".join(L), encoding="utf-8")


PARTIAL = REPORTS_DIR / "partial_results.json"


def main():
    """Stages are checkpointed to reports/partial_results.json; re-running resumes after the
    last finished stage (delete that file, or pass --fresh, to retrain everything)."""
    import sys
    results = {} if ("--fresh" in sys.argv or not PARTIAL.exists()) else json.loads(PARTIAL.read_text(encoding="utf-8"))

    def checkpoint():
        PARTIAL.write_text(json.dumps(results, indent=2), encoding="utf-8")

    p, prep_p = load_pima()
    d, prep_d = load_diabd()
    prep = {"pima": prep_p, "diabd": prep_d}

    if "pima" not in results:
        run_structured("pima", p[PIMA_FEATURES], p["Outcome"].values, results); checkpoint()
    if "diabd" not in results:
        run_structured("diabd", d[DIABD_FEATURES], d["diabetic"].values, results); checkpoint()

    if "robustness_general" not in results or not (MODELS_DIR / "general_structured.joblib").exists():
        c = load_combined()
        fitted, split = run_structured("general", c[GENERAL_FEATURES], c["target"].values, results)
        _, Xte, _, yte = split
        src = c.loc[Xte.index, "source"].values
        results["general_per_source_test_auc"] = {
            s: {m: round(float(roc_auc_score(yte[src == s], fitted[m][0].predict_proba(Xte[src == s])[:, 1])), 4) for m in MANDATORY}
            for s in ["pima", "diabd"]}
        missingness_robustness(fitted, split, ["age", "bmi", "glucose", "systolic_bp", "diastolic_bp"], results, "robustness_general")
        Xtr, _, ytr, _ = split
        pw = float((ytr == 0).sum() / (ytr == 1).sum())
        gx = xgb(pw)
        Xa, ya = mask_augment(Xtr.reset_index(drop=True), ytr, GENERAL_FEATURES, 0.2)
        gx.fit(Xa, ya)
        joblib.dump({"models": {m: fitted[m][0] for m in MANDATORY} | {"XGBoost": gx},
                     "thresholds": {m: fitted[m][1] for m in MANDATORY}, "features": GENERAL_FEATURES},
                    MODELS_DIR / "general_structured.joblib")
        checkpoint()

    if "missing_value_strategy_comparison_pima" not in results:
        missing_strategy_comparison(results); checkpoint()
    if "nlp_and_hybrid_diabd" not in results:
        run_nlp_and_hybrid(results); checkpoint()
    if "hypertension_diabd" not in results:
        run_hypertension(results); checkpoint()

    (REPORTS_DIR / "metrics.json").write_text(json.dumps({"preprocessing": prep, **results}, indent=2), encoding="utf-8")
    write_markdown(results, prep)
    print("Saved models to", MODELS_DIR, "and reports to", REPORTS_DIR)


if __name__ == "__main__":
    main()
