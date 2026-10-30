"""Évalue un cache de lab/physio avec post.run (argmax / quota / doublons+quota). Usage (depuis RMYU) :
python -u lab/physio/post_phys.py <nom>  (lit lab/physio/cache_<nom>.npz) ; ajouter 'mix:<nom2>:<w>' pour mélanger
avec RMYU/cache_<nom2>.npz (poids w sur le physio)."""
import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../.."))
import numpy as np
os.chdir("lab/physio")
from post import run
name = sys.argv[1]
if len(sys.argv) > 2 and sys.argv[2].startswith("mix:"):
    _, other, w = sys.argv[2].split(":"); w = float(w)
    A = np.load(f"cache_{name}.npz"); B = np.load(f"../../cache_{other}.npz")
    out = {}
    for k in A.files:
        out[k] = A[k] if k.startswith("va") else w * A[k] + (1 - w) * B[k]
    for f in range(5):
        assert (A[f"va{f}"] == B[f"va{f}"]).all()
    nm = f"{name}_x_{other}_{w}"; np.savez(f"cache_{nm}.npz", **out); name = nm
run(name)
