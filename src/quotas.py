"""Quotas par personne + sport : la cible est un découpage en trois de chaque couple utilisateur + sport.

Sur train + test réunis, un groupe de N séances contient exactement :
    easy = N // 3      moderate = 2 * N // 3 - N // 3      hard = N - 2 * N // 3
En retirant ce qu'on voit dans le train, on sait combien de séances de chaque classe restent dans le test, par groupe.

On répartit ensuite ces places entre les séances du groupe pour maximiser la somme des log-probabilités du modèle
(méthode hongroise). Seuls des labels du train sont utilisés.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

KEYS = ["user_id", "sport"]


def group_quotas(X_train, y_train, X_test):
    """Nombre de séances de chaque classe restant dans le test, par groupe."""
    seen = pd.crosstab([X_train[k].to_numpy() for k in KEYS], pd.Series(y_train).to_numpy())
    seen.index.names = KEYS
    total = pd.concat([X_train[KEYS], X_test[KEYS]]).groupby(KEYS).size()
    seen = seen.reindex(total.index, fill_value=0)
    return pd.DataFrame({"easy": total // 3 - seen["easy"],
                         "moderate": 2 * total // 3 - total // 3 - seen["moderate"],
                         "hard": total - 2 * total // 3 - seen["hard"]})


def assign_with_quotas(proba, classes, X_test, quotas):
    """Choisit la classe de chaque séance en respectant les quotas de son groupe."""
    classes = np.asarray(classes)
    pred = np.empty(len(X_test), dtype=object)
    for key, idx in X_test.reset_index(drop=True).groupby(KEYS).indices.items():
        slots = np.repeat(np.arange(len(classes)), quotas.loc[key, classes].to_numpy().clip(min=0))
        rows, cols = linear_sum_assignment(-np.log(proba[idx][:, slots] + 1e-9))
        pred[idx[rows]] = classes[slots[cols]]
    return pred
