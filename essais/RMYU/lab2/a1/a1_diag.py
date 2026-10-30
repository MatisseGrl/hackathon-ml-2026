exec(open("lab2/a1/a1_route.py", encoding="utf-8").read().split("def run(")[0])
E = O[:, 1] + 2 * O[:, 2]; r = yi - E
for c in ["r1_res", "r2_res", "r3_res", "r1_z_speed_kmh", "r3_z_speed_kmh"]:
    x = F[c].values[:ntr]; m = ~np.isnan(x)
    print(c, "n", int(m.sum()), "corr(x,y) %.3f" % np.corrcoef(x[m], yi[m])[0, 1], "corr(x,E) %.3f" % np.corrcoef(x[m], E[m])[0, 1],
          "corr(x, y-E) %.3f" % np.corrcoef(x[m], r[m])[0, 1], flush=True)
