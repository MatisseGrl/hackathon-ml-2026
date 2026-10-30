import sys, time, numpy as np, trans as T, stack as S
C = dict(np.load(sys.argv[1])); t0 = time.time()
ref = T.evaluate(C, None); T.rep("ref dup(w4=.5)+quota", ref)
def show(name, r):
    T.rep(name, r); d = r[:, 0] - ref[:, 0]
    print(f"{'':55s} diff vs ref {d.mean():+.4f} (folds +: {(d>0).sum()}/5)", flush=True)
for cfg in [dict(groups=("dh",), lam=0.3), dict(groups=("d",), lam=0.3), dict(groups=("m1",), lam=0.3),
            dict(groups=("ms",), lam=0.3), dict(groups=("dh",), lam=1.0, mode="log"), dict(groups=("dh",), lam=0.1)]:
    show(str(cfg), T.evaluate(C, cfg))
    if cfg["groups"] == ("dh",) and cfg["lam"] == 0.3:
        show(str(cfg) + " pool=test", T.evaluate(C, cfg, test_only_pool=True))
print(f"({time.time()-t0:.0f}s)", flush=True)
for name, kw in [("stack LR logP seul", dict(groups=(), use_lab=False)),
                 ("stack +non étiq.", dict(use_lab=False)),
                 ("stack +étiquetés seuls", dict(use_lab=True, unl=False)),
                 ("stack +non étiq. +étiquetés", dict(use_lab=True))]:
    r, _ = S.run(C, **kw); show(name, r)
print(f"({time.time()-t0:.0f}s)", flush=True)
