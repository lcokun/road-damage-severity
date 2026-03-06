from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = PROJECT_ROOT / "outputs"
EDA_OUT = OUT / "eda"


def main():
    df = pd.read_csv(OUT / "metadata_all.csv")

    country_counts = df["country"].value_counts()
    ax = country_counts.plot(kind="bar")
    ax.set_title("Country Distribution (Annotated Train Data)")
    ax.set_xlabel("Country")
    ax.set_ylabel("Count")
    plt.tight_layout()
    EDA_OUT.mkdir(parents=True, exist_ok=True)
    plt.savefig(EDA_OUT / "country_distribution.png", dpi=200)
    plt.close()

    pivot = pd.pivot_table(df, index="country", columns="final_class", aggfunc="size", fill_value=0)
    pivot = pivot.reindex(columns=["normal", "minor", "severe"])
    ax = pivot.plot(kind="bar", stacked=True)
    ax.set_title("Country vs Class Distribution")
    ax.set_xlabel("Country")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(EDA_OUT / "country_vs_class_stacked.png", dpi=200)
    plt.close()

    print("[OK] Saved plots to", EDA_OUT)


if __name__ == "__main__":
    main()