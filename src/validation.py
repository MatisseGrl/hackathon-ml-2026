"""Validation commune de l'équipe : mêmes folds, même graine, macro-F1.

Utilisation :
    from src.validation import evaluate, plot_confusion, make_submission
    res = evaluate(model, X, y)
    plot_confusion(y, res["oof_pred"])
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
N_SPLITS = 5
LABELS = ["easy", "moderate", "hard"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data(data_dir=DATA_DIR):
    """Charge train, test et sample_submission depuis data/."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    return train, test, sample


def get_folds(y):
    """Toujours les mêmes folds stratifiés pour toute l'équipe."""
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


def make_preprocessor(scale=True):
    """Prétraitement générique : numériques imputées (+ standardisées), catégorielles en one-hot.

    scale=True pour les modèles sensibles à l'échelle (logistique, kNN, SVM),
    scale=False suffit pour les arbres.
    """
    num_steps = [SimpleImputer(strategy="median")]
    if scale:
        num_steps.append(StandardScaler())
    return ColumnTransformer(
        [
            ("num", make_pipeline(*num_steps), make_column_selector(dtype_include="number")),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                ),
                make_column_selector(dtype_exclude="number"),
            ),
        ]
    )


def evaluate(model, X, y, verbose=True):
    """Validation croisée stratifiée. Renvoie la macro-F1 par fold et les prédictions hors fold."""
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    oof_pred = np.empty(len(y), dtype=object)
    scores = []
    for fold, (tr, va) in enumerate(get_folds(y).split(X, y)):
        m = clone(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[va])
        oof_pred[va] = pred
        scores.append(f1_score(y.iloc[va], pred, average="macro"))
        if verbose:
            print(f"fold {fold}: macro-F1 = {scores[-1]:.4f}")
    res = {"scores": scores, "mean": float(np.mean(scores)), "std": float(np.std(scores)), "oof_pred": oof_pred}
    if verbose:
        print(f"==> macro-F1 = {res['mean']:.4f} ± {res['std']:.4f}")
    return res


def plot_confusion(y, oof_pred, labels=LABELS, title="Matrice de confusion (hors fold)"):
    """Matrice de confusion normalisée par ligne + rapport par classe (pour la Gate 2)."""
    print(classification_report(y, oof_pred, labels=labels, digits=3))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ConfusionMatrixDisplay.from_predictions(y, oof_pred, labels=labels, ax=axes[0], colorbar=False)
    axes[0].set_title("Effectifs")
    ConfusionMatrixDisplay.from_predictions(
        y, oof_pred, labels=labels, normalize="true", values_format=".2f", ax=axes[1], colorbar=False
    )
    axes[1].set_title("Proportions par vraie classe")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def make_submission(model, X_train, y_train, X_test, ids, path):
    """Réentraîne sur tout le train, prédit le test et écrit workout_id,intensity."""
    m = clone(model).fit(X_train, y_train)
    sub = pd.DataFrame({"workout_id": ids, "intensity": m.predict(X_test)})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    print(f"{path} écrit ({len(sub)} lignes)")
    print(sub["intensity"].value_counts(normalize=True).round(3))
    return sub
