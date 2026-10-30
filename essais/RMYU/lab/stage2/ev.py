"""Évaluation rapide d'un cache (variantes quota et dupfull+quota de post.py seulement). Usage : python ev.py <nom>..."""
import sys, time; sys.path.append("../.."); sys.path.append(".")
import numpy as np
from sklearn.metrics import f1_score
from src.validation import load_data
from quota import constrained_predict
from post import keyof, adjust, FULL, PART

train, test, _ = load_data()
yi = train.intensity.map({"easy": 0, "moderate": 1, "hard": 2}).values
kf_tr, kp_tr, kf_te, kp_te = keyof(train, FULL), keyof(train, PART), keyof(test, FULL), keyof(test, PART)
VAR = {"quota": dict(w4=0, wfull=0, share=False), "dupfull+quota": dict(w4=0, wfull=1.0), "dup(w4=.5)+quota": dict(w4=0.5)}


def ev(C, variants=("quota", "dup(w4=.5)+quota")):
    res = {k: [] for k in ("argmax",) + tuple(variants)}
    for f in range(5):
        va = C[f"va{f}"]; P = C[f"P{f}"]; nv = len(va)
        lab = np.setdiff1d(np.arange(len(train)), va)
        ku_f = np.concatenate([kf_tr[va], kf_te]); ku_p = np.concatenate([kp_tr[va], kp_te])
        gu = np.concatenate([train.user_id.values[va], test.user_id.values])
        res["argmax"].append(f1_score(yi[va], P[:nv].argmax(1), average="macro"))
        for k in variants:
            Q = adjust(P, ku_f, ku_p, kf_tr[lab], kp_tr[lab], yi[lab], **VAR[k])
            pred = constrained_predict(Q, gu, train.user_id.values[lab], yi[lab])
            res[k].append(f1_score(yi[va], pred[:nv], average="macro"))
    return {k: (np.mean(v), np.std(v), np.round(v, 4).tolist()) for k, v in res.items()}


if __name__ == "__main__":
    for n in sys.argv[1:]:
        t0 = time.time(); r = ev(np.load(n if n.endswith(".npz") else f"cache_{n}.npz"))
        print(n, " | ".join(f"{k} {m:.4f}±{s:.4f}" for k, (m, s, _) in r.items()), f"({time.time()-t0:.0f}s)", flush=True)
