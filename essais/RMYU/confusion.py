"""Matrice de confusion hors-pli du modèle soumis (mélange 0,6 PersonStack + 0,4 Final2, doublons exacts + quotas)."""
import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from src.validation import load_data
from quota import constrained_predict
import post, blend
train, test, _ = load_data()
yi = train.intensity.map({"easy": 0, "moderate": 1, "hard": 2}).values
C = blend.mix(["person", "f2"], [0.6, 0.4])
kf_tr, kf_te, kp_tr, kp_te = post.keyof(train, post.FULL), post.keyof(test, post.FULL), post.keyof(train, post.PART), post.keyof(test, post.PART)
oof = np.zeros(len(train), int); s = []
for f in range(5):
    va = C[f"va{f}"]; lab = np.setdiff1d(np.arange(len(train)), va); nv = len(va)
    Q = post.adjust(C[f"P{f}"], np.concatenate([kf_tr[va], kf_te]), np.concatenate([kp_tr[va], kp_te]), kf_tr[lab], kp_tr[lab], yi[lab], w4=0, wfull=1.0)
    pred = constrained_predict(Q, np.concatenate([train.user_id.values[va], test.user_id.values]), train.user_id.values[lab], yi[lab])
    oof[va] = pred[:nv]; s.append(f1_score(yi[va], pred[:nv], average="macro"))
print("macro-F1 par pli", np.round(s, 4), "moyenne", round(np.mean(s), 4), "±", round(np.std(s), 4))
print(confusion_matrix(yi, oof))
print(classification_report(yi, oof, target_names=["easy", "moderate", "hard"], digits=3))
