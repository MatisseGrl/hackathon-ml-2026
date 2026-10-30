import zlib, os, sys
G = ".git/objects"
def obj(sha):
    p = os.path.join(G, sha[:2], sha[2:])
    if not os.path.exists(p): return None
    d = zlib.decompress(open(p, "rb").read()); i = d.index(b"\0"); return d[:i].decode(), d[i+1:]
def tree(data):
    out = {}; i = 0
    while i < len(data):
        j = data.index(b"\0", i); mode, name = data[i:j].decode().split(" ", 1); out[name] = data[j+1:j+21].hex(); i = j + 21
    return out
def blob_at(commit, path):
    o = obj(commit)
    if o is None: return "packed-commit"
    t = o[1].decode().split("\n")[0].split()[1]
    for part in path.split("/"):
        o = obj(t)
        if o is None: return "packed-tree"
        t = tree(o[1]).get(part)
        if t is None: return "absent"
    return t
old, new = sys.argv[1], sys.argv[2]
for f in ["src/person_stack.py", "src/stacking.py", "src/features.py"]:
    a, b = blob_at(old, f), blob_at(new, f); print(f, a, b, "IDENTIQUE" if a == b else "DIFFERENT")
    if a != b and obj(a) and obj(b):
        import difflib
        A = obj(a)[1].decode("utf-8").splitlines(); B = obj(b)[1].decode("utf-8").splitlines()
        for l in list(difflib.unified_diff(A, B, lineterm="", n=0))[:60]: print("   ", l[:200])
