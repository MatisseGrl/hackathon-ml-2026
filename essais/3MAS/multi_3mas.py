"""Mélange de plusieurs caches avec des poids donnés, doublons + quotas utilisateur+sport ; option --submit chemin.csv."""
import sys
from post_3mas import load, run
names = sys.argv[1].split(","); ws = [float(x) for x in sys.argv[2].split(",")]
sub = sys.argv[sys.argv.index("--submit") + 1] if "--submit" in sys.argv else None
m, s = run(load(names, ws), "user_sport", "floor", True, submit=sub)
print(f"{dict(zip(names, ws))}: {m:.4f} ± {s:.4f}" + (f" -> {sub}" if sub else ""), flush=True)
