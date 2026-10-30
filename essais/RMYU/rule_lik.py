"""Règle d'arrondi des tiers : quand N_total (train+test) n'est pas multiple de 3, quelle classe reçoit l'unité en plus ?
Log-vraisemblance hypergéométrique des comptes du train de chaque user sous chaque hypothèse (tirage sans remise du train
parmi les N_total séances dont les comptes par classe valent T). N'utilise que les comptes du train (pas de fuite)."""
import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from math import lgamma as gammaln
from src.validation import load_data
train, test, _ = load_data()
y = train.intensity.map({"easy": 0, "moderate": 1, "hard": 2})
K = pd.crosstab(train.user_id, y).reindex(columns=[0, 1, 2], fill_value=0)
N = train.user_id.value_counts().add(test.user_id.value_counts(), fill_value=0).reindex(K.index).astype(int)
lc = lambda n, k: gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
for r in (1, 2):
    users = N.index[N % 3 == r]
    opts = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] if r == 1 else [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    print(f"N ≡ {r} (mod 3) : {len(users)} users")
    for o in opts:
        ll = 0.0
        for u in users:
            T = N[u] // 3 + np.array(o); k = K.loc[u].values
            if (k > T).any(): ll = -np.inf; break
            ll += sum(lc(T[c], k[c]) for c in range(3))
        print(f"  unités en plus {o} (easy, moderate, hard) : log-vraisemblance relative {ll:.2f}")
# même chose par simple comptage : excédent moyen du train par classe (attendu n_train/N * T)
for r in (0, 1, 2):
    users = N.index[N % 3 == r]
    nt = K.loc[users].sum(1)
    exc = (K.loc[users].values - (nt.values / N[users].values)[:, None] * (N[users].values[:, None] / 3)).sum(0)
    print(f"N ≡ {r} : excédent cumulé du train par classe vs N/3 (easy, moderate, hard) {np.round(exc, 1)}")
