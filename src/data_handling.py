from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from config import (
    COLUMN_ALIASES,
    ENRON_CSV,
    LABEL_MAP_MALICIOUS,
    LABEL_MAP_SAFE,
    LINGSPAM_CSV,
    PROCESSED_DIR,
)

URL_RE = re.compile(
    r"https?\s*:\s*/\s*/\s*(?:[a-zA-Z0-9-]+\s*\.\s*)+[a-zA-Z]{2,}(?:\s*/\s*[^\s\"'<>]*)?",
    re.IGNORECASE
)
HTML_TAG_RE = re.compile(r"<[^>]+>")
DIGIT_RE = re.compile(r"\d")
SPECIAL_CHAR_RE = re.compile(r"[!$%^&*()_+\-=\[\]{};:\"\\|,.<>/?`~@#]")
EXCLAIM_RE = re.compile(r"!")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class DataQualityReport:
    steps: list = field(default_factory=list)

    def log(self, step: str, **stats):
        self.steps.append({"step": step, **stats})

    def to_json(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.steps, f, indent=2, default=str)

    def print_summary(self):
        for s in self.steps:
            print(f"\n[{s['step']}]")
            for k, v in s.items():
                if k != "step":
                    print(f"  {k}: {v}")


def load_raw_csv(path: Path, source_name: str, dqr: DataQualityReport) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected '{source_name}' CSV at {path}, but it does not exist.\n"
            f"Place your file there (or update config.py paths) and re-run."
        )

    df = pd.read_csv(path)
    n_raw = len(df)

    # TODO: Clean up
    col_subject = "subject"
    col_body = "body"
    col_label = "label"

    # * Usefull?
    missing = [name for name, col in
               [("subject", col_subject), ("body", col_body), ("label", col_label)]
               if col is None]
    if missing:
        raise KeyError(
            f"Could not find column(s) {missing} in {path.name}. "
            f"Available columns: {list(df.columns)}. "
        )

    out = pd.DataFrame({
        "subject": df[col_subject],
        "body": df[col_body],
        "label_raw": df[col_label],
        "source": source_name,
    })

    dqr.log(
        f"load_{source_name}",
        n_records=n_raw,
        n_missing_subject=int(out["subject"].isna().sum()),
        n_missing_body=int(out["body"].isna().sum()),
        n_missing_label=int(out["label_raw"].isna().sum()),
        raw_columns=list(df.columns),
    )
    return out


def normalize_label(value) -> float:
    if pd.isna(value):
        return np.nan
    key = value
    if isinstance(value, str):
        key = value.strip().lower()
    if key in LABEL_MAP_MALICIOUS or value in LABEL_MAP_MALICIOUS:
        return 1.0
    if key in LABEL_MAP_SAFE or value in LABEL_MAP_SAFE:
        return 0.0
    return np.nan


def strip_html(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return HTML_TAG_RE.sub(" ", text)


def clean_text_basic(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = strip_html(text)
    text = text.replace("\xa0", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def combine_and_clean(dfs: list[pd.DataFrame], dqr: DataQualityReport) -> pd.DataFrame:
    df = pd.concat(dfs, ignore_index=True)
    n_before_concat = len(df)
    
    df["has_subject"] = df["subject"].notna() & (df["subject"].astype(str).str.strip() != "")
    df["has_body"] = df["body"].notna() & (df["body"].astype(str).str.strip() != "")
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")

    df["label"] = df["label_raw"].apply(normalize_label)
    n_unlabeled = int(df["label"].isna().sum())
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    df["subject_clean"] = df["subject"].apply(clean_text_basic)
    df["body_clean"] = df["body"].apply(clean_text_basic)

    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=["subject_clean", "body_clean"], keep="first").copy()
    n_after_dedup = len(df)

    df = df.reset_index(drop=True)
    df["email_id"] = df.index.map(lambda i: f"email_{i:06d}")

    dqr.log(
        "combine_and_clean",
        n_before_concat=n_before_concat,
        n_dropped_unlabeled=n_unlabeled,
        n_before_dedup=n_before_dedup,
        n_after_dedup=n_after_dedup,
        n_duplicates_removed=n_before_dedup - n_after_dedup,
        dedup_rate_pct=round(100 * (n_before_dedup - n_after_dedup) / n_before_dedup, 2)
            if n_before_dedup else 0.0,
    )
    return df

def extract_urls(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    
    clean_urls = []
    for match in URL_RE.finditer(text):
        raw_url = match.group(0)
        clean_url = re.sub(r"\s+", "", raw_url)
        clean_urls.append(clean_url)
        
    return clean_urls

def extract_domain(url: str) -> str | None:
    candidate = url if "://" in url else f"http://{url}"
    try:
        netloc = urlparse(candidate).netloc.lower()
        return netloc or None
    except ValueError:
        return None


def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    subj = df["subject_clean"]
    body = df["body_clean"]
    full_text = subj + " " + body

    df["subject_length"] = subj.str.len()
    df["subject_word_count"] = subj.str.split().apply(len)
    df["body_length"] = body.str.len()
    df["body_word_count"] = body.str.split().apply(len)

    urls_per_email = full_text.apply(extract_urls)
    df["num_urls"] = urls_per_email.apply(len)
    domains_per_email = urls_per_email.apply(
        lambda urls: {d for d in (extract_domain(u) for u in urls) if d}
    )
    df["num_unique_domains"] = domains_per_email.apply(len)
    df["has_ip_url"] = urls_per_email.apply(
        lambda urls: any(re.search(r"://\d{1,3}(\.\d{1,3}){3}", u) for u in urls)
    )

    df["num_digits"] = full_text.apply(lambda t: len(DIGIT_RE.findall(t)))
    df["num_special_chars"] = full_text.apply(lambda t: len(SPECIAL_CHAR_RE.findall(t)))
    df["num_exclamations"] = full_text.apply(lambda t: len(EXCLAIM_RE.findall(t)))

    def upper_ratio(t: str) -> float:
        letters = [c for c in t if c.isalpha()]
        if not letters:
            return 0.0
        return sum(1 for c in letters if c.isupper()) / len(letters)

    df["subject_upper_ratio"] = subj.apply(upper_ratio)
    df["body_upper_ratio"] = body.apply(upper_ratio)

    df["digit_density"] = df["num_digits"] / df["body_length"].replace(0, np.nan)
    df["special_char_density"] = df["num_special_chars"] / df["body_length"].replace(0, np.nan)
    df[["digit_density", "special_char_density"]] = df[
        ["digit_density", "special_char_density"]
    ].fillna(0.0)

    return df


def run_data_handling(save: bool = True) -> tuple[pd.DataFrame, DataQualityReport]:
    dqr = DataQualityReport()

    enron = load_raw_csv(ENRON_CSV, "enron", dqr)
    lingspam = load_raw_csv(LINGSPAM_CSV, "lingspam", dqr)

    df = combine_and_clean([enron, lingspam], dqr)
    df = add_structural_features(df)

    cols = [
        "email_id", "source", "label",
        "subject", "body", "subject_clean", "body_clean",
        "has_subject", "has_body",
        "subject_length", "subject_word_count",
        "body_length", "body_word_count",
        "num_urls", "num_unique_domains", "has_ip_url",
        "num_digits", "num_special_chars", "num_exclamations",
        "subject_upper_ratio", "body_upper_ratio",
        "digit_density", "special_char_density",
    ]
    df = df[cols]

    dqr.log(
        "final_dataset",
        n_records=len(df),
        n_malicious=int((df["label"] == 1).sum()),
        n_safe=int((df["label"] == 0).sum()),
        malicious_share_pct=round(100 * (df["label"] == 1).mean(), 2),
        by_source=df["source"].value_counts().to_dict(),
    )

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = PROCESSED_DIR / "emails_features.csv"
        df.to_csv(out_csv, index=False)
        dqr.to_json(PROCESSED_DIR / "data_quality_report.json")
        print(f"Saved {len(df)} records to {out_csv}")

    dqr.print_summary()
    return df, dqr


if __name__ == "__main__":
    run_data_handling()
