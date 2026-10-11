from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from config import PROCESSED_DIR, RANDOM_STATE

STRUCTURAL_FEATURES = [
    "subject_length", "subject_word_count",
    "body_length", "body_word_count",
    "num_urls", "num_unique_domains",
    "num_digits", "num_special_chars", "num_exclamations",
    "subject_upper_ratio", "body_upper_ratio",
    "digit_density", "special_char_density",
]


def choose_n_clusters(n_samples: int, max_clusters: int = 6) -> int:
    if n_samples < 4:
        return 2
    return max(2, min(max_clusters, int(np.sqrt(n_samples / 2)) + 1))


def build_tfidf(
    texts: pd.Series,
    max_features: int = 5000,
    min_df: int = 5,
) -> tuple[TfidfVectorizer, np.ndarray]:
    min_df_eff = 1 if len(texts) < 50 else min_df
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_df=0.9,
        min_df=min_df_eff,
        max_features=max_features,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            max_df=1.0,
            min_df=1,
            max_features=max_features,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        )
        X = vectorizer.fit_transform(texts)
    return vectorizer, X


def get_top_distinctive_terms(
    X_tfidf,
    vocab: np.ndarray,
    in_mask: np.ndarray,
    top_n: int = 12,
) -> list[str]:
    out_mask = ~in_mask
    if in_mask.sum() == 0 or out_mask.sum() == 0:
        return []
    in_mean = np.asarray(X_tfidf[in_mask].mean(axis=0)).ravel()
    out_mean = np.asarray(X_tfidf[out_mask].mean(axis=0)).ravel()
    ratio = np.log((in_mean + 1e-9) / (out_mean + 1e-9))
    idx = np.argsort(-ratio)[:top_n]
    return vocab[idx].tolist()


def prepare_representations(
    df: pd.DataFrame,
    text_col: str = "body_clean",
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    vectorizer, X_text = build_tfidf(df[text_col], max_features=4000)
    vocab = np.array(vectorizer.get_feature_names_out())

    X_struct = StandardScaler().fit_transform(df[STRUCTURAL_FEATURES].values)

    max_svd = min(50, X_text.shape[1] - 1, X_text.shape[0] - 1)
    if max_svd >= 2:
        X_text_reduced = TruncatedSVD(n_components=max_svd, random_state=RANDOM_STATE).fit_transform(X_text)
    else:
        X_text_reduced = X_text.toarray()
    X_text_reduced = StandardScaler().fit_transform(X_text_reduced)

    reps = {
        "structural_only": X_struct,
        "text_only": X_text_reduced,
        "combined": np.hstack([X_struct, X_text_reduced]),
    }
    return reps, X_text, vocab


def describe_clusters(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    X_tfidf,
    vocab: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    out["cluster"] = cluster_labels
    summary = (
        out.groupby("cluster")
        .agg(
            n=("email_id", "count"),
            malicious_pct=("label", lambda s: 100 * s.mean()),
            safe_pct=("label", lambda s: 100 * (1 - s.mean())),
            enron_pct=("source", lambda s: 100 * (s == "enron").mean()),
            lingspam_pct=("source", lambda s: 100 * (s == "lingspam").mean()),
        )
        .reset_index()
    )

    global_means = out[STRUCTURAL_FEATURES].mean()
    cluster_details = {}
    for cid in sorted(out["cluster"].unique()):
        mask = out["cluster"].values == cid
        sub = out.loc[mask]
        centered = (sub[STRUCTURAL_FEATURES].mean() - global_means).abs().sort_values(ascending=False)
        top_structural = centered.head(5).index.tolist()
        cluster_details[f"cluster_{cid}"] = {
            "n": int(mask.sum()),
            "label_share": out.loc[mask, "label"].value_counts(normalize=True).mul(100).round(2).to_dict(),
            "source_share": out.loc[mask, "source"].value_counts(normalize=True).mul(100).round(2).to_dict(),
            "top_distinctive_terms": get_top_distinctive_terms(X_tfidf, vocab, mask, top_n=15),
            "most_shifted_structural_features": {
                feat: float(sub[feat].mean()) for feat in top_structural
            },
        }
    return summary, cluster_details


def run_clustering_analysis(df: pd.DataFrame, save: bool = True) -> dict:
    reps, X_tfidf, vocab = prepare_representations(df)
    n_clusters = choose_n_clusters(len(df))

    cluster_outputs = {}
    for rep_name, X_rep in reps.items():
        model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(X_rep)
        summary, details = describe_clusters(df, labels, X_tfidf, vocab)
        if save:
            summary.to_csv(PROCESSED_DIR / f"clusters_{rep_name}_summary.csv", index=False)
            with open(PROCESSED_DIR / f"clusters_{rep_name}_details.json", "w") as f:
                json.dump(details, f, indent=2)
        cluster_outputs[rep_name] = {
            "summary": summary,
            "details": details,
        }
    return cluster_outputs


def run_malicious_only_clustering(df: pd.DataFrame, save: bool = True) -> dict:
    mal_df = df[df["label"] == 1].copy()
    if len(mal_df) < 10:
        return {}
    reps, X_tfidf, vocab = prepare_representations(mal_df)
    n_clusters = choose_n_clusters(len(mal_df), max_clusters=5)
    labels = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20).fit_predict(reps["combined"])
    summary, details = describe_clusters(mal_df, labels, X_tfidf, vocab)
    if save:
        summary.to_csv(PROCESSED_DIR / "clusters_malicious_only_summary.csv", index=False)
        with open(PROCESSED_DIR / "clusters_malicious_only_details.json", "w") as f:
            json.dump(details, f, indent=2)
    return {"summary": summary, "details": details}
