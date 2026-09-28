#!/usr/bin/env python
"""He Stage 1 -- download He et al. 2020 (Mendeley 29ntw7sh4r) into /workspace/ext/he/raw.

The file list and sha256 come from the public Mendeley API.  The download endpoint is flaky
(it sometimes returns a small JSON error body with HTTP 200), so every file is checked against
its sha256 and retried with backoff.  Re-runnable: files whose sha256 already matches are skipped.

usage:  python external/download_he.py                        # all 68 sections (~0.76 GB)
        python external/download_he.py --subtypes HER2_luminal HER2_non_luminal
writes: raw/<files>, raw/metadata.csv, raw/MANIFEST.tsv (filename, bytes, sha256, ok)
exit 1 if any file is still bad after all retries (rerun; the good ones are kept)
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import he_common as H

API = f"https://data.mendeley.com/public-api/datasets/{H.MENDELEY_ID}"
UA = {"User-Agent": "Mozilla/5.0 (st_benching he download)"}


def get(url, timeout=120):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(f, retries):
    name, cd = f["filename"], f["content_details"]
    dst = H.HE_RAW / name
    if dst.exists() and sha256(dst) == cd["sha256_hash"]:
        return name, f["size"], cd["sha256_hash"], True, "cached"
    err = ""
    for i in range(retries):
        try:
            data = get(cd["download_url"])
            if hashlib.sha256(data).hexdigest() == cd["sha256_hash"]:
                tmp = dst.with_suffix(dst.suffix + ".part")
                tmp.write_bytes(data)
                tmp.replace(dst)
                return name, len(data), cd["sha256_hash"], True, f"try {i + 1}"
            err = f"sha mismatch ({len(data)} B: {data[:60]!r})"
        except Exception as e:                       # noqa: BLE001 -- network errors of any kind
            err = repr(e)
        time.sleep(min(60, 3 * 2 ** i))
    return name, 0, cd["sha256_hash"], False, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtypes", nargs="*", default=None, help="default: all five subtypes")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=8)
    a = ap.parse_args()
    H.HE_RAW.mkdir(parents=True, exist_ok=True)
    meta = json.loads(get(API))
    files = {f["filename"]: f for f in meta["files"]}
    print(f"Mendeley {H.MENDELEY_ID} v{meta.get('version')}: {len(files)} files, "
          f"{sum(f['size'] for f in files.values()) / 1e9:.2f} GB")
    (H.HE_RAW / "dataset.json").write_text(json.dumps(meta, indent=1))

    r = fetch(files["metadata.csv"], a.retries)
    if not r[3]:
        raise SystemExit(f"metadata.csv failed: {r[4]}")
    md = H.metadata()
    if a.subtypes:
        md = md[md["type"].isin(a.subtypes)]
    want = sorted({v for c in ("count_matrix", "histology_image", "spot_coordinates", "tumor_annotation")
                   for v in md[c]})
    missing = [w for w in want if w not in files]
    if missing:
        raise SystemExit(f"metadata.csv names files the API does not list: {missing}")
    print(f"{len(md)} sections / {md.patient.nunique()} patients -> {len(want)} files")

    rows = [r]
    with ThreadPoolExecutor(a.workers) as ex:
        for res in ex.map(lambda n: fetch(files[n], a.retries), want):
            rows.append(res)
            print(("OK  " if res[3] else "BAD ") + f"{res[0]}  {res[4]}", flush=True)
    man = pd.DataFrame(rows, columns=["filename", "bytes", "sha256", "ok", "note"])
    man.to_csv(H.HE_RAW / "MANIFEST.tsv", sep="\t", index=False)
    bad = man[~man.ok]
    print(f"\n{int(man.ok.sum())}/{len(man)} files verified, {man.bytes.sum() / 1e9:.2f} GB")
    if len(bad):
        print("STILL BAD (rerun the script):", list(bad.filename))
        sys.exit(1)


if __name__ == "__main__":
    main()
