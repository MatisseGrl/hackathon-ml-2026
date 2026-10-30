import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np
from sklearn.metrics import f1_score
from src.validation import load_data
tr, te, _ = load_data(); yi = tr.intensity.map({"easy":0,"moderate":1,"hard":2}).values
O = np.load("oof_f2_base.npz")["oof"]
for tag in ["base", "core", "corecap"]:
    C = np.load(f"lab/detective/cache_det_{tag}.npz"); H = np.zeros((len(tr), 3))
    for f in range(5): H[C[f"va{f}"]] = C[f"P{f}"][:len(C[f"va{f}"])]
    out = []
    for w in [0, .2, .3, .4, .5, 1]:
        P = (1-w)*np.log(O+1e-9) + w*np.log(H+1e-9)
        out.append(f"w={w}:{f1_score(yi, P.argmax(1), average='macro'):.4f}")
    print(tag, " ".join(out))
