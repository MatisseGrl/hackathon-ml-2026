"""Mélange de caches (moyenne pondérée des probas), puis évaluation post.py (argmax / quota / doublons+quota).
Usage : python blend.py nomA nomB [poids_A ...]  -> évalue chaque poids ; --submit chemin.csv --w 0.6 pour écrire la soumission."""
import sys; sys.path.append("../.."); sys.path.append(".")
import numpy as np
import post

def mix(names, w):
    C = [np.load(f"cache_{n}.npz") for n in names]
    out = {}
    for k in C[0].files:
        if k.startswith("va"):
            out[k] = C[0][k]
            assert all((c[k] == C[0][k]).all() for c in C)
        else:
            out[k] = sum(wi * c[k] for wi, c in zip(w, C))
    return out

if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    names = [x for x in a if not x.replace(".", "").isdigit()][:2]
    if "--submit" in sys.argv:
        w = float(sys.argv[sys.argv.index("--w") + 1])
        np.savez("cache_blend.npz", **mix(names, [w, 1 - w])); post.run("blend", sys.argv[sys.argv.index("--submit") + 1])
    else:
        for w in [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.0]:
            np.savez("cache_blend.npz", **mix(names, [w, 1 - w]))
            r = post.run("blend", verbose=False)
            print(f"w({names[0]})={w:.1f} " + " | ".join(f"{k} {np.mean(v):.4f}" for k, v in r.items()), flush=True)
