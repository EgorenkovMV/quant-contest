#!/usr/bin/env python3
"""Restore the original binary input files from MANIFEST.json + parts/.

Usage:  python3 tools/reassemble.py [--out DIR] [--check-only]

Binary inputs exceeded the upload path's request-size limit, so they are
stored base64-encoded in parts/<name>/gNNN/part-NNNNN. This script
concatenates and decodes them, verifies sha256, and writes the originals:
  {docx}
  本地调试参考-0818/example/mini_sample/linear.pt
  本地调试参考-0818/example/mini_sample/attn.pt
"""
import argparse, base64, hashlib, json, os, sys

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=here)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    man = json.load(open(os.path.join(here, "MANIFEST.json")))
    ok = True
    for f in man["files"]:
        if f["encoding"] != "base64-parts":
            print(f"ok (whole file): {f['path']}"); continue
        names = []
        for g in sorted(os.listdir(os.path.join(a.out, f["parts_dir"]))):
            names += [os.path.join(g, p) for p in sorted(os.listdir(os.path.join(a.out, f["parts_dir"], g)))]
        data = b"".join(base64.b64decode(open(os.path.join(a.out, f["parts_dir"], n), "rb").read()) for n in names)
        good = len(data) == f["size"] and hashlib.sha256(data).hexdigest() == f["sha256"]
        ok &= good
        print(f"{'ok' if good else 'FAIL'}: {f['path']} ({len(data)} bytes)")
        if not a.check_only and good and not os.path.exists(os.path.join(a.out, f["path"])):
            os.makedirs(os.path.dirname(os.path.join(a.out, f["path"])) or ".", exist_ok=True)
            open(os.path.join(a.out, f["path"]), "wb").write(data)
    sys.exit(0 if ok else 1)

main()
