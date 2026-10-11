from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from clustering import STRUCTURAL_FEATURES, get_top_distinctive_terms, prepare_representations
from config import PROCESSED_DIR, RANDOM_STATE


def run_outlier_analysis(df: pd.DataFrame, save: bool = True) -> dict:
    reps, X_tfidf, vocab = prepare_representations(df)
    X = reps["combined"]
    model = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=RANDOM_STATE,
    )
    flags = model.fit_predict(X)
    scores = model.score_samples(X)
    out = df.copy()
    out["is_outlier"] = (flags == -1).astype(int)
    out["outlier_score"] = scores

    summary = (
        out.groupby("is_outlier")
        .agg(
            n=("email_id", "count"),
            malicious_pct=("label", lambda s: 100 * s.mean()),
            safe_pct=("label", lambda s: 100 * (1 - s.mean())),
            enron_pct=("source", lambda s: 100 * (s == "enron").mean()),
            lingspam_pct=("source", lambda s: 100 * (s == "lingspam").mean()),
        )
        .reset_index()
    )
    if save:
        summary.to_csv(PROCESSED_DIR / "outlier_summary.csv", index=False)

    structural = (
        out.groupby("is_outlier")[STRUCTURAL_FEATURES]
        .mean()
        .T
        .rename(columns={0: "normal_mean", 1: "outlier_mean"})
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    if "normal_mean" not in structural.columns:
        structural["normal_mean"] = np.nan
    if "outlier_mean" not in structural.columns:
        structural["outlier_mean"] = np.nan
    structural["abs_shift"] = (structural["outlier_mean"] - structural["normal_mean"]).abs()
    structural = structural.sort_values("abs_shift", ascending=False)
    if save:
        structural.to_csv(PROCESSED_DIR / "outlier_structural_comparison.csv", index=False)

    outlier_terms = get_top_distinctive_terms(X_tfidf, vocab, out["is_outlier"].values == 1, top_n=20)
    outlier_details = {
        "top_terms_outliers_vs_normal": outlier_terms,
        "malicious_outlier_rate_pct": round(
            100 * ((out["is_outlier"] == 1) & (out["label"] == 1)).sum() / max((out["label"] == 1).sum(), 1), 2
        ),
        "safe_outlier_rate_pct": round(
            100 * ((out["is_outlier"] == 1) & (out["label"] == 0)).sum() / max((out["label"] == 0).sum(), 1), 2
        ),
    }

    mal_mask = (out["is_outlier"] == 1) & (out["label"] == 1)
    safe_mask = (out["is_outlier"] == 1) & (out["label"] == 0)
    if mal_mask.sum() > 2 and safe_mask.sum() > 2:
        outlier_details["malicious_outlier_terms"] = get_top_distinctive_terms(
            X_tfidf, vocab, mal_mask.values, top_n=12
        )
        outlier_details["safe_outlier_terms"] = get_top_distinctive_terms(
            X_tfidf, vocab, safe_mask.values, top_n=12
        )

    if save:
        with open(PROCESSED_DIR / "outlier_details.json", "w") as f:
            json.dump(outlier_details, f, indent=2)
        out.to_csv(PROCESSED_DIR / "emails_with_outlier_flags.csv", index=False)
    return {
        "summary": summary,
        "structural_comparison": structural,
        "details": outlier_details,
    }
