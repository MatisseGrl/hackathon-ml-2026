import sys, os
os.chdir(r"C:\Users\curie\hk\essais\RMYU")
sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
import blendn as B
T = pd.read_pickle("lab2/a5/T.pkl")
fold = T.fold.values; y = B.yi
kf, kp = B.kf_tr, B.kp_tr
full_m = np.zeros(len(T), bool); part_lab = np.full(len(T), np.nan); part_n = np.zeros(len(T))
for f in range(5):
    lab = fold != f; v = np.where(fold == f)[0]
    sf = set(kf[lab])
    full_m[v] = [k in sf for k in kf[v]]
    dfp = pd.DataFrame(dict(k=kp[lab], y=y[lab])).groupby("k").y.agg(["mean", "size"])
    m = pd.Series(kp[v]).map(dfp["mean"]).values; c = pd.Series(kp[v]).map(dfp["size"]).fillna(0).values
    part_lab[v] = m; part_n[v] = c
po = (~full_m) & (part_n > 0)
print("full dup rows", full_m.sum(), "err", T.err[full_m].mean().round(3))
print("partial-only rows", po.sum(), "model err", T.err[po].mean().round(3), "copy-label acc", (np.round(part_lab[po]) == y[po]).mean().round(3))
d = np.abs(part_lab[po] - y[po])
print("partial-only |meanlab - y| dist", np.bincount(np.round(d).astype(int)))
# per-cache error per segment: E rank corr within group (Spearman-like via pairwise concordance)
