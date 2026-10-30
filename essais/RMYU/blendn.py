"""Mélange de N caches (moyenne pondérée des probas) + doublons exacts + quotas par user, sur les 5 plis communs.
- grille de poids (pas 0,1, somme 1), évaluée en parallèle
- sélection imbriquée : pour chaque pli k, on choisit les poids sur les 4 autres plis et on note le score du pli k
  (estimation sans biais de sélection), comparée à la soumission 3 (0,6 person + 0,4 f2).
Usage (depuis essais/RMYU) : python -u blendn.py person f2 neighbor person7 [--step 0.1] [--rule frac|qcut|pct]
La recopie des doublons exacts et le partage des probas entre doublons sont linéaires en P : on les applique
à chaque cache une fois, puis on mélange (identique à post.adjust appliqué au mélange)."""
import sys, itertools, time
sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from src.validation import load_data
from post import keyof, adjust, FULL, PART
from quota import assign

train, test, sample = load_data()
yi = train.intensity.map({"easy": 0, "moderate": 1, "hard": 2}).values
kf_tr, kp_tr, kf_te, kp_te = keyof(train, FULL), keyof(train, PART), keyof(test, FULL), keyof(test, PART)


def targets(n_total, rule):
    """Nombre total de séances de chaque classe pour un user de n_total séances (train+test)."""
    k, r = divmod(n_total, 3)
    extra = {"frac": [[0, 0, 0], [1, 0, 0], [1, 1, 0]],   # règle actuelle de quota.remaining_counts
             "qcut": [[0, 0, 0], [1, 0, 0], [1, 0, 1]],   # pd.qcut(x, 3)
             "pct":  [[0, 0, 0], [0, 0, 1], [0, 1, 1]]}[rule][r]  # coupe du rang en pourcentage
    return k + np.array(extra)


def group_keys(by):
    """Clé de groupe des quotas : "user" ou "user_sport"."""
    if by == "user":
        return train.user_id.values, test.user_id.values
    return keyof(train, ["user_id", "sport"]), keyof(test, ["user_id", "sport"])


def plan(gu, gl, yl, rule):
    """Pour chaque groupe non étiqueté : (lignes, comptes restants)."""
    known = pd.crosstab(pd.Series(gl), pd.Series(yl)).reindex(columns=[0, 1, 2], fill_value=0)
    nl = pd.Series(gl).value_counts()
    out = []
    s = pd.Series(gu)
    for g, rows in s.groupby(s).groups.items():
        rows = np.asarray(rows)
        kc = known.loc[g].values if g in known.index else np.zeros(3, int)
        rem = np.clip(targets(len(rows) + nl.get(g, 0), rule) - kc, 0, None)
        while rem.sum() > len(rows):  # sécurité (ne doit pas arriver)
            rem[np.argmax(rem)] -= 1
        while rem.sum() < len(rows):
            rem[np.argmin(rem)] += 1
        out.append((rows, rem))
    return out


def predict(P, pl):
    pred = P.argmax(1).copy()
    for rows, rem in pl:
        pred[rows] = assign(P[rows], rem)
    return pred


def prepare(names, rule="frac", by="user", oracle=False):
    """oracle=True : quotas sur la validation seule, avec ses comptes réels par groupe (situation de la soumission,
    où les comptes restants du test sont connus exactement) ; sinon quotas sur validation + test (comptes estimés)."""
    C = {n: np.load(f"cache_{n}.npz") for n in names}
    va = [C[names[0]][f"va{f}"] for f in range(5)]
    for n in names:
        assert all((C[n][f"va{f}"] == va[f]).all() for f in range(5)), n
    folds = []
    for f in range(5):
        lab = np.setdiff1d(np.arange(len(train)), va[f])
        ku_f = np.concatenate([kf_tr[va[f]], kf_te]); ku_p = np.concatenate([kp_tr[va[f]], kp_te])
        gtr, gte = group_keys(by)
        gu = np.concatenate([gtr[va[f]], gte])
        Q = {n: adjust(C[n][f"P{f}"], ku_f, ku_p, kf_tr[lab], kp_tr[lab], yi[lab], w4=0, wfull=1.0) for n in names}
        R = {n: C[n][f"P{f}"] for n in names}
        if oracle:
            nv = len(va[f]); g = pd.Series(gtr[va[f]])
            pl = [(np.asarray(r), np.bincount(yi[va[f]][r], minlength=3)) for r in g.groupby(g).indices.values()]
            Q = {n: q[:nv] for n, q in Q.items()}
        else:
            pl = plan(gu, gtr[lab], yi[lab], rule)
        folds.append(dict(va=va[f], Q=Q, R=R, pl=pl))
    return folds


def score(folds, names, w):
    s, a = [], []
    for fd in folds:
        nv = len(fd["va"])
        Q = sum(wi * fd["Q"][n] for wi, n in zip(w, names))
        R = sum(wi * fd["R"][n] for wi, n in zip(w, names))
        s.append(f1_score(yi[fd["va"]], predict(Q, fd["pl"])[:nv], average="macro"))
        a.append(f1_score(yi[fd["va"]], R[:nv].argmax(1), average="macro"))
    return np.array(s), np.array(a)


def grid(m, step):
    n = int(round(1 / step))
    for c in itertools.product(range(n + 1), repeat=m - 1):
        if sum(c) <= n:
            yield np.array(list(c) + [n - sum(c)]) * step


def nested(S):
    """S : (n_combos, 5) scores par pli -> score imbriqué par pli et indices choisis."""
    out, idx = [], []
    for k in range(5):
        other = [j for j in range(5) if j != k]
        i = int(np.argmax(S[:, other].mean(1)))
        out.append(S[i, k]); idx.append(i)
    return np.array(out), idx


if __name__ == "__main__":
    args = [x for x in sys.argv[1:]]
    step = float(args[args.index("--step") + 1]) if "--step" in args else 0.1
    rule = args[args.index("--rule") + 1] if "--rule" in args else "frac"
    names = [x for x in args if not x.startswith("--") and x not in (str(step), rule)]
    t0 = time.time()
    folds = prepare(names, rule)
    W = list(grid(len(names), step))
    print(f"{names} règle={rule} : {len(W)} combinaisons, préparation {time.time()-t0:.0f}s", flush=True)
    res = Parallel(n_jobs=10)(delayed(score)(folds, names, w) for w in W)
    S = np.array([r[0] for r in res]); A = np.array([r[1] for r in res])
    m = S.mean(1)
    order = np.argsort(-m)
    for i in order[:12]:
        print(" ".join(f"{n}={w:.1f}" for n, w in zip(names, W[i])),
              f"dup+quota {m[i]:.4f} ± {S[i].std():.4f} | argmax {A[i].mean():.4f} | plis", np.round(S[i], 4), flush=True)
    nest, idx = nested(S)
    print(f"imbriqué : {nest.mean():.4f} ± {nest.std():.4f} plis {np.round(nest, 4)} ; choix",
          [" ".join(f"{w:.1f}" for w in W[i]) for i in idx], flush=True)
    np.savez(f"grid_{'_'.join(names)}_{rule}.npz", W=np.array(W), S=S, A=A)
    print(f"fini {time.time()-t0:.0f}s", flush=True)
