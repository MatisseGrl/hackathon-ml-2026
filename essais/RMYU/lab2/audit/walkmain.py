import sys, time; sys.argv = [sys.argv[0], "x", "y"]
exec(open("essais/RMYU/lab2/audit/packread.py").read().split("old, new = sys.argv")[0])
m = get("9a7bc0b345bc0a6c4b173b7e6ba68b7d3766124a")[1].decode().split("\n")
parents = [l.split()[1] for l in m if l.startswith("parent ")]; print("parents", parents)
c = parents[1]
for _ in range(25):
    txt = get(c)[1].decode(); lines = txt.split("\n")
    ts = [l for l in lines if l.startswith("committer ")][0].split()[-2]
    msg = txt.split("\n\n", 1)[1].strip().split("\n")[0][:70]
    print(time.strftime("%H:%M:%S", time.localtime(int(ts))), c[:8], "PS", (blob(c, "src/person_stack.py") or "None")[:8], "ST", (blob(c, "src/stacking.py") or "None")[:8], msg)
    ps = [l.split()[1] for l in lines if l.startswith("parent ")]
    if not ps: break
    c = ps[0]
