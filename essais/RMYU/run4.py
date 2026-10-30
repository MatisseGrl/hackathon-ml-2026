import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from src.validation import load_data, get_folds
from feats import build, add_resid
from knnlab import KnnLabel
from quota import constrained_predict
train,test,_ = load_data()
F = add_resid(build(train,test))
X = F.drop(columns=["part","workout_id"]).reset_index(drop=True)
ntr = (F.part=="tr").sum(); Xtr, Xte = X.iloc[:ntr].reset_index(drop=True), X.iloc[ntr:].reset_index(drop=True)
y = train.intensity.reset_index(drop=True); yi = y.map({"easy":0,"moderate":1,"hard":2}).values
model = KnnLabel(cols=("resid_spd","ldist","t"))
res = {k: [] for k in ["argmax","q_us","q_u"]}
for tr_i, va_i in get_folds(y).split(Xtr, y):
    m = clone(model).fit(Xtr.iloc[tr_i], y.iloc[tr_i])
    Xu = pd.concat([Xtr.iloc[va_i], Xte]).reset_index(drop=True)
    P = m.predict_proba(Xu)
    nv = len(va_i)
    res["argmax"].append(f1_score(yi[va_i], P[:nv].argmax(1), average="macro"))
    for name, keyf in [("q_us", lambda d: (d.user_id.astype(str)+"_"+d.sport.astype(str)).values), ("q_u", lambda d: d.user_id.values)]:
        pr = constrained_predict(P, keyf(Xu), keyf(Xtr.iloc[tr_i]), yi[tr_i])
        res[name].append(f1_score(yi[va_i], pr[:nv], average="macro"))
for k, v in res.items(): print(k, round(np.mean(v),4), "±", round(np.std(v),4))
