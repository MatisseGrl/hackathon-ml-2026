"""Application au test en soumission (exemple ; écrit seulement dans lab2/a4).
Q_test = moyenne égale des Pfull des 5 caches, doublons exacts recopiés (post.adjust), puis lissage transductif
vers les séances de TEST voisines (seules non étiquetées en soumission), puis quotas exacts par (user, sport)."""
import sys; sys.path.append("../.."); sys.path.append("."); sys.path.append("lab2/a4")
import numpy as np, pandas as pd
import blendn as B
from post import adjust, LAB
from trans import level_keys, smooth

NAMES = ["person", "f2", "neighbor", "person7", "resid2"]


def transduce_test(alpha=0.0, wts=None, beta=0.0, names=NAMES):
    tr, te = B.train, B.test; ntr = len(tr)
    P = sum(np.load(f"cache_{n}.npz")["Pfull"] for n in names) / len(names)
    Q = adjust(P, B.kf_te, B.kp_te, B.kf_tr, B.kp_tr, B.yi, w4=0, wfull=1.0)
    fixed = Q.max(1) > 0.999
    if alpha != 0 or beta != 0:
        keys = level_keys(pd.concat([tr, te], ignore_index=True))
        rows = ntr + np.arange(len(te))
        Q = smooth(Q, keys, rows, rows, Q, np.arange(ntr), B.yi, wts or {"slot": 1}, alpha, beta, fixed)
        Q = np.clip(Q, 1e-6, None); Q /= Q.sum(1, keepdims=True)
    gtr, gte = B.group_keys("user_sport")
    pl = B.plan(gte, gtr, B.yi, "pct")   # comptes restants exacts du test par (user, sport)
    return B.predict(Q, pl)


if __name__ == "__main__":
    p0 = transduce_test(0.0)
    p1 = transduce_test(0.2, {"slot": 1})
    p2 = transduce_test(0.2, {"route": 1})
    print("changements vs référence : S a=0.2 ->", (p1 != p0).mean().round(4), "; R a=0.2 ->", (p2 != p0).mean().round(4))
    print("comptes réf", np.bincount(p0, minlength=3), "S", np.bincount(p1, minlength=3))
    pd.DataFrame({"workout_id": B.test.workout_id.values, "intensity": LAB[p1]}).to_csv("lab2/a4/sub_a4_S02_exemple.csv", index=False)
    print("écrit lab2/a4/sub_a4_S02_exemple.csv (exemple, NON recommandé)")
