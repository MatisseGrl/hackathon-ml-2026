"""Variantes de mélange : moyenne géométrique (log-probabilités) et « température » (P ** a) avant les quotas."""
import sys
import numpy as np
from post_3mas import run
from pathlib import Path
HERE = Path(__file__).resolve().parent

def mix(names, ws, geo=False, temp=1.0):
    C = [np.load(HERE / f"cache/{n}.npz") for n in names]
    out = {k: C[0][k] for k in C[0].files if k.startswith("va")}
    for k in C[0].files:
        if k.startswith("va"):
            continue
        if geo:
            L = sum(w * np.log(np.clip(c[k], 1e-9, 1)) for w, c in zip(ws, C)) * temp
        else:
            L = np.log(np.clip(sum(w * c[k] for w, c in zip(ws, C)), 1e-9, 1)) * temp
        P = np.exp(L - L.max(1, keepdims=True))
        out[k] = P / P.sum(1, keepdims=True)
    return out

names = sys.argv[1].split(","); ws = [float(x) for x in sys.argv[2].split(",")]
for geo in (False, True):
    for temp in (0.5, 1.0, 2.0):
        m, s = run(mix(names, ws, geo, temp), "user_sport", "floor", True)
        print(f"{names} {ws} geo={geo} temp={temp}: {m:.4f} ± {s:.4f}", flush=True)
