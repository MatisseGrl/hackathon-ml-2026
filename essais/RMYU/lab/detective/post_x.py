"""Variante de post.py : clé de doublon partiel restreinte au même mois (accord 0,865 vs 0,35-0,5 sinon).
Usage (depuis RMYU) : python -u lab/detective/post_x.py <chemin_cache.npz>"""
import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from src.validation import load_data
from quota import constrained_predict
from post import adjust, keyof, FULL, PART
PARTM = PART + ["year", "month"]
PARTG = PART + ["year", "month", "altitude_gain_m"]
train, test, _ = load_data(); yi = train.intensity.map({"easy": 0, "moderate": 1, "hard": 2}).values
C = np.load(sys.argv[1])
kf_tr, kf_te = keyof(train, FULL), keyof(test, FULL)
res = {}
for pname, pk in [("PART", PART), ("PART+month", PARTM), ("PART+month+gain", PARTG)]:
    kp_tr, kp_te = keyof(train, pk), keyof(test, pk)
    for w4 in ([0.0] if pname != "PART" else [0.0, 0.5]) + ([0.5, 0.7, 0.9] if pname != "PART" else []):
        sc = []
        for f in range(5):
            va = C[f"va{f}"]; P = C[f"P{f}"]; nv = len(va); lab = np.setdiff1d(np.arange(len(train)), va)
            Q = adjust(P, np.concatenate([kf_tr[va], kf_te]), np.concatenate([kp_tr[va], kp_te]), kf_tr[lab], kp_tr[lab], yi[lab], w4=w4)
            pred = constrained_predict(Q, np.concatenate([train.user_id.values[va], test.user_id.values]), train.user_id.values[lab], yi[lab])
            sc.append(f1_score(yi[va], pred[:nv], average="macro"))
        print(f"{pname:18s} w4={w4:.1f}  {np.mean(sc):.4f} ± {np.std(sc):.4f}", flush=True)
