import zlib, os, glob, struct, sys, difflib
G = ".git/objects"
packs = []
for idx in glob.glob(G + "/pack/*.idx"):
    d = open(idx, "rb").read(); assert d[:4] == b"\xfftOc"
    fan = struct.unpack(">256I", d[8:8+1024]); n = fan[-1]
    shas = d[1032:1032+20*n]; off0 = 1032 + 20*n + 4*n; offs = struct.unpack(f">{n}I", d[off0:off0+4*n])
    big0 = off0 + 4*n
    packs.append((idx[:-4] + ".pack", fan, shas, offs, d, big0, n))
def find(sha):
    b = bytes.fromhex(sha)
    for pk, fan, shas, offs, d, big0, n in packs:
        lo = fan[b[0]-1] if b[0] else 0; hi = fan[b[0]]
        while lo < hi:
            m = (lo+hi)//2; s = shas[20*m:20*m+20]
            if s == b:
                o = offs[m]
                if o & 0x80000000: o = struct.unpack(">Q", d[big0+8*(o & 0x7fffffff):big0+8*(o & 0x7fffffff)+8])[0]
                return pk, o
            if s < b: lo = m+1
            else: hi = m
    return None
def delta(base, dl):
    i = 0
    def varint():
        nonlocal i; r = 0; s = 0
        while True:
            c = dl[i]; i += 1; r |= (c & 0x7f) << s; s += 7
            if not c & 0x80: return r
    varint(); varint(); out = bytearray()
    while i < len(dl):
        c = dl[i]; i += 1
        if c & 0x80:
            o = sz = 0
            for k in range(4):
                if c & (1 << k): o |= dl[i] << (8*k); i += 1
            for k in range(3):
                if c & (1 << (4+k)): sz |= dl[i] << (8*k); i += 1
            out += base[o:o+(sz or 0x10000)]
        else:
            out += dl[i:i+c]; i += c
    return bytes(out)
def read_at(pk, off):
    f = open(pk, "rb"); f.seek(off); c = f.read(1)[0]; t = (c >> 4) & 7; s = 4
    while c & 0x80: c = f.read(1)[0]; s += 7
    if t == 6:
        c = f.read(1)[0]; r = c & 0x7f
        while c & 0x80: c = f.read(1)[0]; r = ((r + 1) << 7) | (c & 0x7f)
        bt, base = read_at(pk, off - r); dl = zlib.decompressobj().decompress(f.read(10**7)); return bt, delta(base, dl)
    if t == 7:
        bs = f.read(20).hex(); bt, base = get(bs); dl = zlib.decompressobj().decompress(f.read(10**7)); return bt, delta(base, dl)
    return {1: "commit", 2: "tree", 3: "blob"}.get(t, str(t)), zlib.decompressobj().decompress(f.read(10**7))
def get(sha):
    p = os.path.join(G, sha[:2], sha[2:])
    if os.path.exists(p):
        d = zlib.decompress(open(p, "rb").read()); i = d.index(b"\0"); return d[:i].decode().split()[0], d[i+1:]
    loc = find(sha); return read_at(*loc)
def tree(data):
    out = {}; i = 0
    while i < len(data):
        j = data.index(b"\0", i); name = data[i:j].decode().split(" ", 1)[1]; out[name] = data[j+1:j+21].hex(); i = j + 21
    return out
def blob(commit, path):
    t = get(commit)[1].decode().split("\n")[0].split()[1]
    for part in path.split("/"):
        t = tree(get(t)[1]).get(part)
        if t is None: return None
    return t
old, new = sys.argv[1], sys.argv[2]
for p in ["src/person_stack.py", "src/stacking.py", "src/features.py"]:
    a, b = blob(old, p), blob(new, p); print("==", p, a, b, "IDENTIQUE" if a == b else "DIFFERENT")
    if a and b and a != b:
        A = get(a)[1].decode().splitlines(); B = get(b)[1].decode().splitlines()
        for l in list(difflib.unified_diff(A, B, lineterm="", n=0))[:80]: print("   ", l[:180])
