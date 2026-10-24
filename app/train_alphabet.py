import argparse
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ASL alphabet classifier.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/samples.csv"),
        help="CSV file produced by collect_alphabet.py",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/alphabet_clf.joblib"),
        help="Where to store the trained model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise SystemExit(f"Dataset {path} not found. Run collect_alphabet.py first.")
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise SystemExit("Dataset missing 'label' column.")
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    return X, y


def main() -> None:
    args = parse_args()
    X, y = load_dataset(args.input)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    label_counts = pd.Series(y).value_counts()
    too_small = label_counts[label_counts < 2]
    stratify_param: Optional[np.ndarray] = y
    if not too_small.empty:
        print(
            "Warning: some classes have <2 samples; disabling stratified split for now."
        )
        print(too_small)
        stratify_param = None

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=args.test_size, stratify=stratify_param, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("Classification report:")
    print(classification_report(y_val, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, y_pred))

    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "scaler": scaler}, args.model)
    print(f"Saved model to {args.model}")


if __name__ == "__main__":
    main()

