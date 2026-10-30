import sys, time, numpy as np, trans as T
C = dict(np.load(sys.argv[1])); t0 = time.time()
pt = "--pt" in sys.argv
R = {}
def go(name, cfg, **kw):
    R[name] = T.rep(name, T.evaluate(C, cfg, test_only_pool=pt, **kw))
go("ref dup+quota", None)
for g in [("dh",), ("d",), ("ms",), ("m",), ("m1",)]:
    for lam in [0.3, 1.0]:
        go(f"{g} lam={lam}", dict(groups=g, lam=lam))
print(f"({time.time()-t0:.0f}s)", flush=True)
for g in [("dh", "m1"), ("d", "m1"), ("dh", "ms")]:
    for lam in [0.3, 1.0]:
        go(f"{g} lam={lam}", dict(groups=g, lam=lam))
for g in [("dh",), ("m1",)]:
    go(f"{g} lam=1 log", dict(groups=g, lam=1.0, mode="log"))
    go(f"{g} lam=1 pseudo=.5", dict(groups=g, lam=1.0, pseudo=0.5))
    go(f"{g} lam=1 lamL=.3", dict(groups=g, lam=1.0, lamL=0.3))
    go(f"{g} lam=1 iters=3", dict(groups=g, lam=1.0, iters=3))
print(f"({time.time()-t0:.0f}s)", flush=True)
