"""
Utility script to synthesize a larger training set from the collected seed
samples and export a pretrained classifier for immediate use.

Usage:
    python app/build_pretrained.py \
        --input data/samples.csv \
        --output data/pretrained_samples.csv \
        --model models/alphabet_clf.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a pretrained ASL classifier.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/samples.csv"),
        help="Seed samples (one or more per class).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pretrained_samples.csv"),
        help="Where to store the augmented dataset.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/alphabet_clf.joblib"),
        help="Path for the pretrained model bundle.",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=400,
        help="How many synthetic samples to create per label.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.01,
        help="Standard deviation of Gaussian noise added during augmentation.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Validation split size for sanity checking.",
    )
    return parser.parse_args()


def augment_samples(df: pd.DataFrame, per_class: int, noise_std: float) -> pd.DataFrame:
    features = df.columns[1:]
    augmented_rows = []
    rng = np.random.default_rng(42)

    for label, group in df.groupby("label"):
        source = group[features].values
        for _ in range(per_class):
            base = source[rng.integers(0, len(source))]
            noise = rng.normal(0, noise_std, size=base.shape)
            augmented_rows.append([label, *(base + noise)])

    augmented_df = pd.DataFrame(augmented_rows, columns=df.columns)
    return augmented_df.sample(frac=1.0, random_state=42).reset_index(drop=True)


def train_and_save(df: pd.DataFrame, model_path: Path, test_size: float) -> None:
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=1, n_jobs=-1, random_state=42
    )
    clf.fit(X_train, y_train)
    acc = clf.score(X_val, y_val)
    print(f"Validation accuracy on synthetic set: {acc:.3f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "scaler": scaler}, model_path)
    print(f"Wrote pretrained model to {model_path}")


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Seed dataset not found: {args.input}")
    seed_df = pd.read_csv(args.input)
    augmented_df = augment_samples(seed_df, args.per_class, args.noise_std)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(args.output, index=False)
    print(
        f"Augmented dataset with {len(augmented_df)} rows written to {args.output} "
        f"({args.per_class} per class)."
    )
    train_and_save(augmented_df, args.model, args.test_size)


if __name__ == "__main__":
    main()

