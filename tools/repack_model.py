"""Copy a GC model tarball member by member, replacing the named members with local files (e.g. v4.1 = v4 with a new
code/report.py), then re-read the output and check that every member is byte-identical to the input except the
replaced ones, which must equal their files.

usage: python repack_model.py images\\model_v4.tar.gz images\\model_v41.tar.gz code/report.py=code\\container\\report_v41.py
"""
import hashlib
import io
import sys
import tarfile
import time


def digests(path):
    out = {}
    with tarfile.open(path, "r:gz") as t:
        for m in t:
            h = hashlib.sha256()
            f = t.extractfile(m)
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
            out[m.name] = h.hexdigest()
    return out


src, dst, *pairs = sys.argv[1:]
new = dict(p.split("=", 1) for p in pairs)
with tarfile.open(src, "r:gz") as tin, tarfile.open(dst, "w:gz", compresslevel=6) as tout:
    names = tin.getnames()
    assert all(n in names for n in new), f"not in {src}: {set(new) - set(names)}"
    for m in tin:
        if m.name in new:
            data = open(new[m.name], "rb").read()
            m.size, m.mtime = len(data), int(time.time())
            tout.addfile(m, fileobj=io.BytesIO(data))
        else:
            tout.addfile(m, tin.extractfile(m))
a, b = digests(src), digests(dst)
assert list(a) == list(b), "member list or order changed"
changed = [n for n in a if a[n] != b[n]]
assert set(changed) <= set(new), f"unexpected changes: {set(changed) - set(new)}"
for n, p in new.items():
    assert b[n] == hashlib.sha256(open(p, "rb").read()).hexdigest(), f"{n} does not match {p}"
print(f"REPACK OK {dst}: {len(b)} members, replaced {sorted(new)}, changed {changed}", flush=True)
