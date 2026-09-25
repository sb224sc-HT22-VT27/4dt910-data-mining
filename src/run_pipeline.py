"""
Runs Step 1 (Data Handling) -> Step 2 (Text Mining) end-to-end.

Usage:
    python src/run_pipeline.py

Expects data/raw/enron.csv and data/raw/lingspam.csv to exist
(see config.py for paths / expected column names).
"""

from data_handling import run_data_handling
from text_mining import run_text_mining


def main():
    print("STEP 1: DATA HANDLING")
    df, dqr = run_data_handling(save=True)

    print("\n\nSTEP 2: TEXT MINING")
    results = run_text_mining(df, save=True)

    print("\n\nPipeline complete. Outputs in data/processed/ and figures/.")


if __name__ == "__main__":
    main()
