import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from src.validation import load_data, get_folds
from feats import build, add_resid
from timelab import TimeLabel
from quota import constrained_predict
train,test,_ = load_data()
F = add_resid(build(train,test))
X = F.drop(columns=["part","workout_id"]).reset_index(drop=True)
ntr = (F.part=="tr").sum(); Xtr, Xte = X.iloc[:ntr].reset_index(drop=True), X.iloc[ntr:].reset_index(drop=True)
y = train.intensity.reset_index(drop=True); yi = y.map({"easy":0,"moderate":1,"hard":2}).values
key = lambda d: d.user_id.values
model = TimeLabel()
a, q = [], []
for tr_i, va_i in get_folds(y).split(Xtr, y):
    m = clone(model).fit(Xtr.iloc[tr_i], y.iloc[tr_i])
    Xu = pd.concat([Xtr.iloc[va_i], Xte]).reset_index(drop=True); nv=len(va_i)
    P = m.predict_proba(Xu)
    a.append(f1_score(yi[va_i], P[:nv].argmax(1), average="macro"))
    q.append(f1_score(yi[va_i], constrained_predict(P, key(Xu), key(Xtr.iloc[tr_i]), yi[tr_i])[:nv], average="macro"))
    print(len(a), round(a[-1],4), round(q[-1],4), flush=True)
print("argmax", round(np.mean(a),4), "| quota", round(np.mean(q),4), "±", round(np.std(q),4))
