from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = PROJECT_ROOT / "outputs"
EDA_OUT = OUT / "eda"


def plot_split_dist(csv_path: Path, title: str, out_path: Path):
    df = pd.read_csv(csv_path)
    counts = df["final_class"].value_counts().reindex(["normal", "minor", "severe"])
    ax = counts.plot(kind="bar")
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_all_splits(train_csv: Path, val_csv: Path, test_csv: Path, out_path: Path):
    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)
    df_te = pd.read_csv(test_csv)

    classes = ["normal", "minor", "severe"]
    tr = df_tr["final_class"].value_counts().reindex(classes).fillna(0)
    va = df_va["final_class"].value_counts().reindex(classes).fillna(0)
    te = df_te["final_class"].value_counts().reindex(classes).fillna(0)

    plot_df = pd.DataFrame({"train": tr, "val": va, "test": te})
    ax = plot_df.plot(kind="bar")
    ax.set_title("Class Distribution Across Splits")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    train_csv = OUT / "metadata_train.csv"
    val_csv = OUT / "metadata_val.csv"
    test_csv = OUT / "metadata_test.csv"

    plot_split_dist(train_csv, "Train Class Distribution", EDA_OUT / "train_class_dist.png")
    plot_split_dist(val_csv, "Validation Class Distribution", EDA_OUT / "val_class_dist.png")
    plot_split_dist(test_csv, "Test Class Distribution", EDA_OUT / "test_class_dist.png")
    plot_all_splits(train_csv, val_csv, test_csv, EDA_OUT / "all_splits_class_dist.png")

    print("[OK] Saved plots to", EDA_OUT)


if __name__ == "__main__":
    main()