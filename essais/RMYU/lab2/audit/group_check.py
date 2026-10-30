import sys; sys.path.append("../../../.."); sys.path.append("../..")
import numpy as np, pandas as pd
from src.validation import load_data, get_folds
train, test, _ = load_data()
y = train.intensity.map({"easy":0,"moderate":1,"hard":2}).values
gtr = (train.user_id.astype(str) + "|" + train.sport.astype(str)).values
gte = (test.user_id.astype(str) + "|" + test.sport.astype(str)).values
def stats(g):
    s = pd.Series(g).value_counts()
    sz = pd.Series(g).map(s).values
    return len(s), np.mean(sz), (sz == 1).mean(), (sz <= 2).mean(), np.median(sz)
print("test   : groupes %d, taille moy (par ligne) %.1f, lignes en groupe de 1 %.3f, <=2 %.3f, mediane %.0f" % stats(gte))
for f, (tr, va) in enumerate(get_folds(y).split(np.zeros(len(y)), y)):
    print(f"val {f}  : groupes %d, taille moy (par ligne) %.1f, lignes en groupe de 1 %.3f, <=2 %.3f, mediane %.0f" % stats(gtr[va]))
# calibration : confiance vs exactitude sur la validation
for n in ["person","person7","f2","neighbor","resid2"]:
    z = np.load(f"../../cache_{n}.npz"); acc=[]; conf=[]
    for f in range(5):
        va = z[f"va{f}"]; Pv = z[f"P{f}"][:len(va)]
        acc.append((Pv.argmax(1) == y[va]).mean()); conf.append(Pv.max(1).mean())
    print(f"{n:9s} exactitude val {np.mean(acc):.4f} vs confiance val {np.mean(conf):.4f}")
