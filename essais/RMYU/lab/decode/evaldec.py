"""Évalue des décodages sur un cache (format cache.py). Usage : python -u lab/decode/evaldec.py <cache.npz> [variantes...]
(lancer depuis C:/Users/curie/hk/essais/RMYU)"""
import sys, time; sys.path.append("../.."); sys.path.append("."); sys.path.append("lab/decode")
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from src.validation import load_data
from quota import constrained_predict
from post import keyof, adjust, FULL, PART
from decode import decode

train, test, sample = load_data()
yi = train.intensity.map({"easy": 0, "moderate": 1, "hard": 2}).values
kf_tr, kp_tr, kf_te, kp_te = keyof(train, FULL), keyof(train, PART), keyof(test, FULL), keyof(test, PART)


def variants():
    V = {"REF quota(post)": ("ref", None, 0)}
    for w4 in [0.0, 0.5]:
        V[f"REF dup w4={w4}"] = ("ref", None, w4)
    import os
    if os.environ.get("FAST"):
        for w4 in [0.3, 0.5, 0.7]:
            for meth, a in [("rank", 1), ("rank_d", 1), ("rank_lr", 1), ("rank_sink", 1), ("rank_sink", 0.7),
                            ("sink_argmax", 0.5), ("sink_argmax", 0.7), ("sink_argmax", 1)]:
                V[f"w4={w4} ceil {meth} a={a}"] = ("ceil", meth, w4, a)
        return V
    for w4 in [0.0, 0.5]:
        for rule in ["orig", "ceil", "mix"]:
            for meth, a in [("lsa_log", 1), ("lsa_lin", 1), ("rank", 1), ("marg_argmax", 1), ("marg_lsa", 1),
                            ("marg_argmax", 0.7), ("marg_argmax", 1.4), ("sink_argmax", 1)]:
                if rule == "mix" and meth.startswith("lsa"):
                    continue
                V[f"w4={w4} {rule} {meth} a={a}"] = (rule, meth, w4, a)
    return V


def run(cache, sel=None):
    C = np.load(cache)
    V = variants()
    if sel:
        V = {k: v for k, v in V.items() if k.startswith("REF") or any(s in k for s in sel)}
    res = {k: [] for k in V}; t0 = time.time()
    for f in range(5):
        va = C[f"va{f}"]; P = C[f"P{f}"]; nv = len(va)
        lab = np.setdiff1d(np.arange(len(train)), va)
        ku_f = np.concatenate([kf_tr[va], kf_te]); ku_p = np.concatenate([kp_tr[va], kp_te])
        gu = np.concatenate([train.user_id.values[va], test.user_id.values])
        gl, yl = train.user_id.values[lab], yi[lab]
        Qcache = {}
        for k, v in V.items():
            if v[0] == "ref":
                if v[1] is None and k == "REF quota(post)":
                    Q = adjust(P, ku_f, ku_p, kf_tr[lab], kp_tr[lab], yl, w4=0, wfull=0, share=False)
                else:
                    Q = adjust(P, ku_f, ku_p, kf_tr[lab], kp_tr[lab], yl, w4=v[2])
                pred = constrained_predict(Q, gu, gl, yl)
            else:
                rule, meth, w4, a = v
                if w4 not in Qcache:
                    Qcache[w4] = adjust(P, ku_f, ku_p, kf_tr[lab], kp_tr[lab], yl, w4=w4)
                pred = decode(Qcache[w4], gu, gl, yl, rule=rule, method=meth, a=a)
            res[k].append(f1_score(yi[va], pred[:nv], average="macro"))
        print(f"fold {f} ({time.time()-t0:.0f}s)", flush=True)
    ref = np.array(res["REF dup w4=0.5"])
    for k, v in res.items():
        v = np.array(v); d = v - ref
        print(f"{k:40s} {v.mean():.4f} ± {v.std():.4f} | diff vs REF dup.5 {d.mean():+.4f} ± {d.std():.4f} (+{(d>0).sum()}/5)")
    return res


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2:] or None)
