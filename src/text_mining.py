from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from config import FIGURES_DIR, PROCESSED_DIR, RANDOM_STATE

STRUCTURAL_FEATURES = [
    "subject_length", "subject_word_count",
    "body_length", "body_word_count",
    "num_urls", "num_unique_domains",
    "num_digits", "num_special_chars", "num_exclamations",
    "subject_upper_ratio", "body_upper_ratio",
    "digit_density", "special_char_density",
]

def descriptive_stats_by_class(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    safe = df[df["label"] == 0]
    malicious = df[df["label"] == 1]

    for feat in STRUCTURAL_FEATURES:
        s_vals = safe[feat].dropna()
        m_vals = malicious[feat].dropna()

        u_stat, p_val = stats.mannwhitneyu(s_vals, m_vals, alternative="two-sided")
        n1, n2 = len(s_vals), len(m_vals)
        effect = 1 - (2 * u_stat) / (n1 * n2) if n1 and n2 else np.nan

        rows.append({
            "feature": feat,
            "safe_mean": s_vals.mean(), "safe_median": s_vals.median(), "safe_std": s_vals.std(),
            "malicious_mean": m_vals.mean(), "malicious_median": m_vals.median(), "malicious_std": m_vals.std(),
            "mannwhitney_p": p_val,
            "rank_biserial_effect": effect,
        })

    out = pd.DataFrame(rows).sort_values("rank_biserial_effect", key=abs, ascending=False)
    return out.reset_index(drop=True)


def plot_structural_feature_comparisons(df: pd.DataFrame, top_n: int = 6, stats_df: pd.DataFrame | None = None):
    feats = (
        stats_df["feature"].head(top_n).tolist()
        if stats_df is not None
        else STRUCTURAL_FEATURES[:top_n]
    )
    n = len(feats)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, feat in zip(axes, feats):
        data = [df.loc[df.label == 0, feat].dropna(), df.loc[df.label == 1, feat].dropna()]

        clip = np.nanpercentile(np.concatenate(data), 99)
        data_clipped = [np.clip(d, None, clip) for d in data]
        ax.boxplot(data_clipped, labels=["Safe", "Malicious"], showfliers=False)
        ax.set_title(feat)
    for ax in axes[len(feats):]:
        ax.axis("off")

    fig.suptitle("Structural feature comparison: Safe vs Malicious (99th pct clipped, outliers hidden)")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "structural_features_safe_vs_malicious.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {path}")


def malicious_share_by_source(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("source")["label"]
        .agg(n="count", malicious_share=lambda s: s.mean())
        .reset_index()
    )


def build_tfidf(texts: pd.Series, max_features: int = 5000) -> tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_df=0.9,
        min_df=5,
        max_features=max_features,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    X = vectorizer.fit_transform(texts)
    return vectorizer, X


def top_discriminative_terms(df: pd.DataFrame, text_col: str = "body_clean",
                              top_n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    vectorizer, X = build_tfidf(df[text_col])
    vocab = np.array(vectorizer.get_feature_names_out())

    y = df["label"].values
    safe_mean = np.asarray(X[y == 0].mean(axis=0)).ravel()
    mal_mean = np.asarray(X[y == 1].mean(axis=0)).ravel()

    eps = 1e-9
    log_ratio_mal = np.log((mal_mean + eps) / (safe_mean + eps))

    order_mal = np.argsort(-log_ratio_mal)[:top_n]
    order_safe = np.argsort(log_ratio_mal)[:top_n]

    top_mal = pd.DataFrame({
        "term": vocab[order_mal],
        "malicious_tfidf_mean": mal_mean[order_mal],
        "safe_tfidf_mean": safe_mean[order_mal],
        "log_ratio_malicious_over_safe": log_ratio_mal[order_mal],
    })
    top_safe = pd.DataFrame({
        "term": vocab[order_safe],
        "malicious_tfidf_mean": mal_mean[order_safe],
        "safe_tfidf_mean": safe_mean[order_safe],
        "log_ratio_malicious_over_safe": log_ratio_mal[order_safe],
    })
    return top_safe, top_mal


def topic_model_subset(texts: pd.Series, n_topics: int = 8, n_top_words: int = 12) -> dict:
    if len(texts) < 20:
        return {}
    vectorizer, X = build_tfidf(texts, max_features=3000)
    vocab = np.array(vectorizer.get_feature_names_out())

    model = NMF(n_components=n_topics, random_state=RANDOM_STATE, init="nndsvda", max_iter=400)
    W = model.fit_transform(X)

    topics = {}
    for i, comp in enumerate(model.components_):
        top_idx = np.argsort(-comp)[:n_top_words]
        topics[f"topic_{i}"] = {
            "top_words": vocab[top_idx].tolist(),
            "weight_share_pct": round(100 * W[:, i].sum() / W.sum(), 2) if W.sum() > 0 else 0.0,
        }
    return topics


def run_text_mining(df: pd.DataFrame, save: bool = True):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


    print("\nPHASE A — Descriptive statistics (safe vs malicious)")
    print("-" * 70)
    stats_df = descriptive_stats_by_class(df)
    print(stats_df.to_string(index=False))
    if save:
        stats_df.to_csv(PROCESSED_DIR / "descriptive_stats.csv", index=False)

    plot_structural_feature_comparisons(df, top_n=6, stats_df=stats_df)

    share_by_source = malicious_share_by_source(df)
    print("\nMalicious share by source corpus:")
    print(share_by_source.to_string(index=False))

    print("\nPHASE B — Vocabulary contrast (TF-IDF log-ratio, malicious vs safe)")
    print("-" * 70)
    top_safe, top_mal = top_discriminative_terms(df)
    print("\nTop terms associated with SAFE emails:")
    print(top_safe.head(15).to_string(index=False))
    print("\nTop terms associated with MALICIOUS emails:")
    print(top_mal.head(15).to_string(index=False))
    if save:
        top_safe.to_csv(PROCESSED_DIR / "top_terms_safe.csv", index=False)
        top_mal.to_csv(PROCESSED_DIR / "top_terms_malicious.csv", index=False)

    print("\nPHASE B — Topic modelling (NMF), run separately per class")
    print("-" * 70)
    safe_texts = df.loc[df.label == 0, "body_clean"]
    mal_texts = df.loc[df.label == 1, "body_clean"]
    topics_safe = topic_model_subset(safe_texts)
    topics_mal = topic_model_subset(mal_texts)

    print("\nSafe-subset topics:")
    for name, t in topics_safe.items():
        print(f"  {name} ({t['weight_share_pct']}%): {', '.join(t['top_words'])}")
    print("\nMalicious-subset topics:")
    for name, t in topics_mal.items():
        print(f"  {name} ({t['weight_share_pct']}%): {', '.join(t['top_words'])}")

    if save:
        with open(PROCESSED_DIR / "topics_safe.json", "w") as f:
            json.dump(topics_safe, f, indent=2)
        with open(PROCESSED_DIR / "topics_malicious.json", "w") as f:
            json.dump(topics_mal, f, indent=2)

    return {
        "descriptive_stats": stats_df,
        "malicious_share_by_source": share_by_source,
        "top_terms_safe": top_safe,
        "top_terms_malicious": top_mal,
        "topics_safe": topics_safe,
        "topics_malicious": topics_mal,
    }


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "emails_features.csv")
    run_text_mining(df)
