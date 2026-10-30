"""Mélange de caches (moyenne pondérée des P{f} et Pfull ; --geo = moyenne des log). Usage (depuis RMYU) :
python -u lab/stage2/blend.py <nom_sortie> <cache1>:<w1> <cache2>:<w2> ... [--geo]
<cacheK> = nom court (-> cache_<nom>.npz dans RMYU) ou chemin vers un .npz. Écrit cache_stage2_<nom_sortie>.npz et l'évalue."""
import sys; sys.path.append("../.."); sys.path.append("."); sys.path.append("lab/stage2")
import numpy as np


def load(n):
    return np.load(n if n.endswith(".npz") else f"cache_{n}.npz")


def blend(items, geo=False):
    C = [(load(n), float(w)) for n, w in items]
    W = sum(w for _, w in C); out = {}
    for k in ["Pfull"] + [f"P{f}" for f in range(5)]:
        if geo:
            L = sum(w * np.log(np.clip(c[k], 1e-6, 1)) for c, w in C) / W
            P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        else:
            P = sum(w * c[k] for c, w in C) / W
        out[k] = P
    for f in range(5):
        va = C[0][0][f"va{f}"]
        assert all((c[f"va{f}"] == va).all() for c, _ in C)
        out[f"va{f}"] = va
    return out


if __name__ == "__main__":
    from ev import ev
    out_name = sys.argv[1]; geo = "--geo" in sys.argv
    items = [a.rsplit(":", 1) for a in sys.argv[2:] if not a.startswith("--")]
    out = blend(items, geo)
    np.savez(f"cache_stage2_{out_name}.npz", **out)
    r = ev(out)
    print("EVAL", out_name, " | ".join(f"{k} {m:.4f}±{s:.4f}" for k, (m, s, _) in r.items()), flush=True)
