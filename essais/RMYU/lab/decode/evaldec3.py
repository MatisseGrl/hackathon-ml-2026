"""Évaluation ciblée : marginales exactes (DP) puis argmax / f1opt. Usage : python -u lab/decode/evaldec3.py <cache.npz>"""
import sys, time; sys.path.append("../.."); sys.path.append("."); sys.path.append("lab/decode")
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from src.validation import load_data
from quota import constrained_predict
from post import keyof, adjust, FULL, PART
from decode import decode, f1opt

train, test, sample = load_data()
yi = train.intensity.map({"easy": 0, "moderate": 1, "hard": 2}).values
kf_tr, kp_tr, kf_te, kp_te = keyof(train, FULL), keyof(train, PART), keyof(test, FULL), keyof(test, PART)
C = np.load(sys.argv[1])
res = {}; t0 = time.time()
def add(k, v): res.setdefault(k, []).append(v)
for f in range(5):
    va = C[f"va{f}"]; P = C[f"P{f}"]; nv = len(va)
    lab = np.setdiff1d(np.arange(len(train)), va)
    ku_f = np.concatenate([kf_tr[va], kf_te]); ku_p = np.concatenate([kp_tr[va], kp_te])
    gu = np.concatenate([train.user_id.values[va], test.user_id.values])
    gl, yl = train.user_id.values[lab], yi[lab]
    sc = lambda pred: f1_score(yi[va], pred[:nv], average="macro")
    for w4 in [0.0, 0.3, 0.5]:
        Q = adjust(P, ku_f, ku_p, kf_tr[lab], kp_tr[lab], yl, w4=w4)
        add(f"REF dup w4={w4} (post quota)", sc(constrained_predict(Q, gu, gl, yl)))
        add(f"w4={w4} ceil lsa_log", sc(decode(Q, gu, gl, yl, rule="ceil", method="lsa_log")))
        for rule in ["orig", "ceil", "mix"]:
            for a in ([0.85, 1.0, 1.2] if rule == "ceil" else [1.0]):
                pred, M = decode(Q, gu, gl, yl, rule=rule, method="marg_argmax", a=a, return_probs=True)
                add(f"w4={w4} {rule} marg_argmax a={a}", sc(pred))
                add(f"w4={w4} {rule} marg_f1opt a={a}", sc(f1opt(M)))
    print(f"fold {f} ({time.time()-t0:.0f}s)", flush=True)
ref = np.array(res["REF dup w4=0.0 (post quota)"])
for k, v in res.items():
    v = np.array(v); d = v - ref
    print(f"{k:36s} {v.mean():.4f} ± {v.std():.4f} | diff vs REF dup w4=0 {d.mean():+.4f} ± {d.std():.4f} (+{(d>0).sum()}/5)")
