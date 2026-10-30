import sys, time
sys.path.append("../.."); sys.path.append(".")
import numpy as np
t0 = time.time()
import blendn as B
y = B.yi
rng = np.random.default_rng(0)
ND = 2000; NS = 3835
def mf1(yy, p):
    c = np.bincount(yy * 3 + p, minlength=9).reshape(3, 3)
    tp = np.diag(c); fp = c.sum(0) - tp; fn = c.sum(1) - tp
    d = 2 * tp + fp + fn
    return np.mean(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0))
def bl(F, w):
    return [B.predict(sum(wi * fd["Q"][n] for n, wi in w.items()), fd["pl"]) for fd in F]
pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
DRAWS = None
def report(name, P5, PX, VA):
    global DRAWS
    yv = [y[v] for v in VA]
    if DRAWS is None:
        DRAWS = []
        for k in range(ND):
            i, j = pairs[rng.integers(len(pairs))]
            n = len(yv[i]) + len(yv[j])
            idx = rng.choice(n, NS, replace=(n < NS))
            DRAWS.append((i, j, idx))
    d = []
    for i, j, idx in DRAWS:
        yy = np.concatenate([yv[i], yv[j]])[idx]
        a = np.concatenate([P5[i], P5[j]])[idx]; b = np.concatenate([PX[i], PX[j]])[idx]
        d.append(mf1(yy, b) - mf1(yy, a))
    d = np.array(d)
    dis = np.mean(np.concatenate([P5[f] != PX[f] for f in range(5)]))
    f5 = np.mean([mf1(yv[f], P5[f]) for f in range(5)]); fx = np.mean([mf1(yv[f], PX[f]) for f in range(5)])
    print(f"RES {name}: desaccord={dis:.4f} F1plis5={f5:.4f} F1plisX={fx:.4f} mean_delta={d.mean():+.5f} sd={d.std():.5f} "
          f"P(X>5)={np.mean(d>0):.3f} P(X=5)={np.mean(d==0):.3f} gain=E[max(0,d)]={np.maximum(d,0).mean():.5f} ({time.time()-t0:.0f}s)", flush=True)
five = ["person", "f2", "neighbor", "person7", "resid2"]
F = B.prepare(five, "pct", "user_sport", oracle=True)
VA = [fd["va"] for fd in F]
print("sizes", [len(v) for v in VA], f"prep {time.time()-t0:.0f}s", flush=True)
P5 = bl(F, {n: 0.2 for n in five})
report("b_n4_person_seul_us", P5, bl(F, {"person": 1.0}), VA)
FU = B.prepare(["person", "f2"], "pct", "user", oracle=True)
assert all((FU[f]["va"] == VA[f]).all() for f in range(5))
report("a_n3_0.6person0.4f2_user", P5, bl(FU, {"person": 0.6, "f2": 0.4}), VA)
F8 = B.prepare(["person11", "f2s7", "neighbor7"], "pct", "user_sport", oracle=True)
for f in range(5):
    F[f]["Q"].update(F8[f]["Q"])
names = five + ["person11", "f2s7", "neighbor7"]
report("c_familles8", P5, bl(F, {"person": 0.4/3, "person7": 0.4/3, "person11": 0.4/3, "f2": 0.1, "f2s7": 0.1,
                                 "neighbor": 0.1, "neighbor7": 0.1, "resid2": 0.2}), VA)
report("d_egal8", P5, bl(F, {n: 1/8 for n in names}), VA)
print("fini", flush=True)
