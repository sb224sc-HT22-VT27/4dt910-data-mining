from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from config import FIGURES_DIR, PROCESSED_DIR, RANDOM_STATE

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


def plot_structural_feature_comparisons(
    df: pd.DataFrame,
    top_n: int = 6,
    stats_df: pd.DataFrame | None = None,
    save: bool = True,
):
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
        ax.boxplot(data_clipped, label=["Safe", "Malicious"], showfliers=False)
        ax.set_title(feat)
    for ax in axes[len(feats):]:
        ax.axis("off")

    fig.suptitle("Structural feature comparison: Safe vs Malicious (99th pct clipped, outliers hidden)")
    fig.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "structural_features_safe_vs_malicious.png"
        fig.savefig(path, dpi=150)
        print(f"Saved figure to {path}")
    plt.close(fig)


def malicious_share_by_source(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("source")["label"]
        .agg(n="count", malicious_share=lambda s: s.mean())
        .reset_index()
    )


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


def descriptive_stats_within_each_source(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for source_name, sub in df.groupby("source"):
        sub_stats = descriptive_stats_by_class(sub).copy()
        sub_stats.insert(0, "source", source_name)
        parts.append(sub_stats)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def cross_source_effect_direction(within_source_stats: pd.DataFrame) -> pd.DataFrame:
    if within_source_stats.empty:
        return pd.DataFrame()
    pivot = within_source_stats.pivot_table(
        index="feature",
        columns="source",
        values="rank_biserial_effect",
        aggfunc="first",
    )
    out = pivot.copy()
    out["direction_consistent_across_sources"] = out.apply(
        lambda row: np.all(np.sign(row.dropna().values) == np.sign(row.dropna().values[:1]))
        if len(row.dropna()) > 1
        else np.nan,
        axis=1,
    )
    out["mean_abs_effect"] = out.abs().mean(axis=1)
    return out.sort_values("mean_abs_effect", ascending=False).reset_index()


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
        n_svd = max_svd
        X_text_reduced = TruncatedSVD(n_components=n_svd, random_state=RANDOM_STATE).fit_transform(X_text)
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
    X_rep = reps["combined"]
    labels = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20).fit_predict(X_rep)
    summary, details = describe_clusters(mal_df, labels, X_tfidf, vocab)
    if save:
        summary.to_csv(PROCESSED_DIR / "clusters_malicious_only_summary.csv", index=False)
        with open(PROCESSED_DIR / "clusters_malicious_only_details.json", "w") as f:
            json.dump(details, f, indent=2)
    return {"summary": summary, "details": details}


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


def label_semantics_report(top_safe: pd.DataFrame, top_mal: pd.DataFrame) -> dict:
    spam_marketing_terms = {"free", "click", "offer", "price", "buy", "win", "money", "marketing", "remove"}
    phishing_terms = {"account", "verify", "password", "bank", "security", "urgent", "login", "confirm"}

    mal_terms = set(top_mal["term"].head(60).str.lower().tolist())
    spam_hits = sorted(mal_terms.intersection(spam_marketing_terms))
    phishing_hits = sorted(mal_terms.intersection(phishing_terms))

    guidance = (
        "Use 'malicious/spam-like' wording for this corpus rather than assuming all positives are phishing."
        if len(spam_hits) >= len(phishing_hits)
        else "Positive labels include phishing-like content; keep wording at least 'malicious/phishing-like'."
    )
    return {
        "top_malicious_terms_sample": sorted(list(mal_terms))[:20],
        "spam_like_hits": spam_hits,
        "phishing_like_hits": phishing_hits,
        "terminology_guidance": guidance,
    }


def topic_model_subset(texts: pd.Series, n_topics: int = 8, n_top_words: int = 12) -> dict:
    if len(texts) < 20:
        return {}
    vectorizer, X = build_tfidf(texts, max_features=3000)
    vocab = np.array(vectorizer.get_feature_names_out())
    n_topics_eff = min(n_topics, X.shape[0], X.shape[1])
    if n_topics_eff < 2:
        return {}

    model = NMF(n_components=n_topics_eff, random_state=RANDOM_STATE, init="nndsvda", max_iter=400)
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

    plot_structural_feature_comparisons(df, top_n=6, stats_df=stats_df, save=save)

    share_by_source = malicious_share_by_source(df)
    print("\nMalicious share by source corpus:")
    print(share_by_source.to_string(index=False))
    if save:
        share_by_source.to_csv(PROCESSED_DIR / "malicious_share_by_source.csv", index=False)

    print("\nPHASE A2 — Safe vs malicious within each source")
    print("-" * 70)
    within_source = descriptive_stats_within_each_source(df)
    source_direction = cross_source_effect_direction(within_source)
    print(within_source.head(20).to_string(index=False))
    print("\nFeatures with consistent direction across sources:")
    print(source_direction.head(15).to_string(index=False))
    if save:
        within_source.to_csv(PROCESSED_DIR / "descriptive_stats_within_source.csv", index=False)
        source_direction.to_csv(PROCESSED_DIR / "descriptive_stats_source_direction.csv", index=False)

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
    terminology = label_semantics_report(top_safe, top_mal)
    print("\nLabel terminology guidance:")
    print(terminology["terminology_guidance"])
    if save:
        with open(PROCESSED_DIR / "label_semantics_report.json", "w") as f:
            json.dump(terminology, f, indent=2)

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

    print("\nPHASE C — Clustering comparisons (structural vs text vs combined)")
    print("-" * 70)
    clustering = run_clustering_analysis(df, save=save)
    for rep_name, rep_result in clustering.items():
        print(f"\n{rep_name} cluster summary:")
        print(rep_result["summary"].to_string(index=False))

    print("\nPHASE C2 — Clustering only malicious emails")
    print("-" * 70)
    malicious_clustering = run_malicious_only_clustering(df, save=save)
    if malicious_clustering:
        print(malicious_clustering["summary"].to_string(index=False))
    else:
        print("Not enough malicious records for malicious-only clustering.")

    print("\nPHASE D — Outlier detection")
    print("-" * 70)
    outlier_results = run_outlier_analysis(df, save=save)
    print(outlier_results["summary"].to_string(index=False))
    print("\nTop shifted structural features among outliers:")
    print(outlier_results["structural_comparison"].head(10).to_string(index=False))
    print("\nOutlier interpretation:")
    print(json.dumps(outlier_results["details"], indent=2))

    return {
        "descriptive_stats": stats_df,
        "malicious_share_by_source": share_by_source,
        "descriptive_stats_within_source": within_source,
        "source_effect_direction": source_direction,
        "top_terms_safe": top_safe,
        "top_terms_malicious": top_mal,
        "label_semantics_report": terminology,
        "topics_safe": topics_safe,
        "topics_malicious": topics_mal,
        "clustering": clustering,
        "malicious_only_clustering": malicious_clustering,
        "outlier_analysis": outlier_results,
    }


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "emails_features.csv")
    run_text_mining(df)
