"""Post-traitements sur des probabilités en cache (cache_proba.py), évalués comme pour le vrai test :
dans chaque pli, les séances de validation + les séances du test sont « non étiquetées » ; seules les étiquettes
du reste du train servent (doublons exacts et quotas).
- mélange de plusieurs caches (moyenne pondérée des probabilités) ;
- doublons exacts (16 colonnes) : on recopie le label du train (essais/RMYU/post.py) ;
- quotas exacts (tiers de train + test) par utilisateur, ou par utilisateur + sport,
  affectation optimale en NumPy pur (essais/RMYU/quota.py, pas de scipy)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path += [str(ROOT), str(ROOT / "essais/RMYU")]
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from src.validation import load_data
from quota import assign, LAB
from post import FULL, keyof

HERE = Path(__file__).resolve().parent
train, test, sample = load_data()
YI = train.intensity.map({"easy": 0, "moderate": 1, "hard": 2}).to_numpy()


def load(names, weights):
    C = [np.load(HERE / f"cache/{n}.npz") for n in names]
    out = {k: C[0][k] for k in C[0].files if k.startswith("va")}
    for k in C[0].files:
        if not k.startswith("va"):
            out[k] = sum(w * c[k] for w, c in zip(weights, C))
    return out


def counts(n, rule):
    """Nombre total de chaque classe dans un groupe de n séances."""
    if rule == "floor":
        return np.array([n // 3, 2 * n // 3 - n // 3, n - 2 * n // 3])
    return np.full(3, n / 3.0)      # « frac » : RMYU, arrondi au plus proche (remaining_counts)


def remaining(n_total, known, m, rule):
    if rule == "floor":
        r = np.clip(counts(n_total, rule) - known, 0, None)
        while r.sum() > m:          # sécurité (ne devrait pas arriver)
            r[r.argmax()] -= 1
        while r.sum() < m:
            r[r.argmin()] += 1
        return r
    from quota import remaining_counts
    return remaining_counts(n_total, known, m)


def quota_predict(P, g_unl, g_lab, y_lab, rule):
    pred = P.argmax(1).copy()
    known = pd.crosstab(pd.Series(g_lab), pd.Series(y_lab)).reindex(columns=[0, 1, 2], fill_value=0)
    n_lab = pd.Series(g_lab).value_counts()
    for g, rows in pd.Series(range(len(g_unl))).groupby(g_unl).groups.items():
        rows = np.asarray(rows)
        kc = known.loc[g].to_numpy() if g in known.index else np.zeros(3, int)
        pred[rows] = assign(P[rows], remaining(len(rows) + n_lab.get(g, 0), kc, len(rows), rule))
    return pred


def dup_copy(P, k_unl, k_lab, y_lab):
    P = P.copy()
    lab = pd.DataFrame(np.eye(3)[y_lab], index=k_lab).groupby(level=0).mean()
    hit = pd.Index(lab.index).get_indexer(k_unl)
    P[hit >= 0] = lab.to_numpy()[hit[hit >= 0]]
    return pd.DataFrame(P).groupby(k_unl).transform("mean").to_numpy()   # doublons non étiquetés : même proba


def group_key(df, by):
    return df["user_id"].astype(str).to_numpy() if by == "user" else (df["user_id"].astype(str) + "|" + df["sport"]).to_numpy()


def run(C, by="user_sport", rule="floor", dup=True, submit=None):
    kf_tr, kf_te = keyof(train, FULL), keyof(test, FULL)
    g_tr, g_te = group_key(train, by), group_key(test, by)
    scores = []
    for f in range(5):
        va, P = C[f"va{f}"], C[f"P{f}"]
        lab = np.setdiff1d(np.arange(len(train)), va)
        if dup:
            P = dup_copy(P, np.concatenate([kf_tr[va], kf_te]), kf_tr[lab], YI[lab])
        if by is None:
            pred = P.argmax(1)
        else:
            pred = quota_predict(P, np.concatenate([g_tr[va], g_te]), g_tr[lab], YI[lab], rule)
        scores.append(f1_score(YI[va], pred[:len(va)], average="macro"))
    if submit:
        P = dup_copy(C["Pfull"], kf_te, kf_tr, YI) if dup else C["Pfull"]
        pred = quota_predict(P, g_te, g_tr, YI, rule) if by else P.argmax(1)
        sub = pd.DataFrame({"workout_id": test.workout_id.to_numpy(), "intensity": LAB[pred]})
        assert len(sub) == len(sample) and (sub.workout_id.to_numpy() == sample.workout_id.to_numpy()).all()
        sub.to_csv(submit, index=False)
    return float(np.mean(scores)), float(np.std(scores))


if __name__ == "__main__":
    names = sys.argv[1].split(",")
    weights = [float(w) for w in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1 / len(names)] * len(names)
    C = load(names, weights)
    for by, rule, dup in [(None, None, False), ("user", "frac", True), ("user", "floor", True),
                          ("user_sport", "frac", True), ("user_sport", "floor", True), ("user_sport", "floor", False)]:
        m, s = run(C, by, rule, dup)
        print(f"{'+'.join(names)} {weights} | quotas {by} {rule} doublons {dup}: {m:.4f} ± {s:.4f}", flush=True)
