"""Contrôle de la règle « pct » (N≡1 : hard +1 ; N≡2 : moderate +1, hard +1) : aucun compte du train ne dépasse sa cible,
et combien de users sont déjà saturés (plus aucune séance de cette classe dans le test)."""
import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from src.validation import load_data
import blendn as B
train, test, _ = load_data()
y = train.intensity.map({"easy": 0, "moderate": 1, "hard": 2})
K = pd.crosstab(train.user_id, y).reindex(columns=[0, 1, 2], fill_value=0)
N = train.user_id.value_counts().add(test.user_id.value_counts(), fill_value=0).reindex(K.index).astype(int)
for rule in ("frac", "qcut", "pct"):
    T = np.array([B.targets(n, rule) for n in N.values])
    over = (K.values > T).sum(0); sat = (K.values == T).sum(0)
    print(f"{rule:5s} users avec train > cible par classe {over} ; saturés (train = cible) {sat}")
print("taille N : min", N.min(), "médiane", int(N.median()), "; part du test par user min/max",
      round((test.user_id.value_counts() / N).min(), 3), round((test.user_id.value_counts() / N).max(), 3))
