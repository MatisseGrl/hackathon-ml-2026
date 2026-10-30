"""Teste des poids de mélange entre caches, avec doublons + quotas utilisateur+sport (règle floor).
Usage : python blend_3mas.py nomA,nomB w1 w2 ...   (poids de nomA, nomB reçoit 1 - w)"""
import sys
from post_3mas import load, run
names = sys.argv[1].split(",")
for w in map(float, sys.argv[2:]):
    C = load(names, [w, 1 - w])
    for by in ("user", "user_sport"):
        m, s = run(C, by, "floor", True)
        print(f"{names[0]}={w:.2f} {names[1]}={1 - w:.2f} quotas {by}: {m:.4f} ± {s:.4f}", flush=True)
