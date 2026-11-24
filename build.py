#!/usr/bin/env python3
"""Stage 2: download, resize, de-duplicate, write the training manifest.

Input : candidates/<label>.jsonl  (from collect.py)
Output: images/<Label>/*.jpg, manifest.csv, manifest.jsonl,
        rejected.jsonl, report.md, classes.json

Re-runs are incremental: files already on disk are counted and their perceptual
hashes are reused, so raising --per-label only fetches the difference.
"""
import argparse, csv, json, os, re, threading, io
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image

Image.MAX_IMAGE_PIXELS = 300_000_000

ROOT = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(ROOT, "species.json"), encoding="utf-8"))
LABELS = [s["label"] for s in CFG["species"]]
META = {s["label"]: s for s in CFG["species"]}

UA_INAT = {"User-Agent": "tree-dataset-builder/0.1 (research image dataset) python-requests"}
UA_COM = {"User-Agent": "tree-dataset-builder/0.1 (https://qoder.com; research image dataset) python-requests"}

NC_KEYS = ("nc", "nc-sa", "nc-nd", "by-nc")
OK_KEYS = ("pd", "cc0", "public domain", "cc-by", "by-sa", "gfdl", "cc by")


def classify_license(code):
    c = (code or "").lower().replace(" ", "-")
    if not c or c in ("none", "unknown", ""):
        return "unknown"
    if "cc-by-nc" in c or c in ("cc-nc", "cc-nc-sa", "cc-nc-nd") or "by-nc" in c:
        return "no"                      # non-commercial only
    if c in ("cc-by", "cc-by-sa", "cc-by-nd", "cc0", "cc-pd", "pd", "public-domain") \
       or re.match(r"^cc-by(-sa)?-[0-9.]+$", c) or "public domain" in c or c.startswith("gfdl"):
        return "yes"
    return "unknown"


def dhash(img, size=8):
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for r in range(size):
        base = r * (size + 1)
        for c in range(size):
            bits = (bits << 1) | (1 if px[base + c] < px[base + c + 1] else 0)
    return bits


def hamming(a, b):
    return bin(a ^ b).count("1")


def commons_thumb(url, long_side):
    from urllib.parse import quote
    if "/commons/" not in url or not re.search(r"\.(jpe?g|png)$", url, re.I):
        return url
    head, path = url.split("/commons/", 1)
    path = quote(path, safe="/")          # filenames often contain spaces / CJK
    fname = path.rsplit("/", 1)[-1]
    return f"{head}/commons/thumb/{path}/{long_side}px-{fname}"


def fetch(url, ua, tries=3):
    for _ in range(tries):
        try:
            r = requests.get(url, headers=ua, timeout=90)
            if r.status_code == 200 and r.content:
                return r.content
            if r.status_code in (403, 429, 500, 502, 503, 504):
                continue
            return None
        except requests.RequestException:
            continue
    return None


class Store:
    """Shared hash set + sequential filename counter, guarded by one lock."""

    def __init__(self, hashes, start_idx):
        self.hashes = hashes
        self.idx = start_idx
        self.lock = threading.Lock()

    def claim(self, h, thresh):
        with self.lock:
            for other in self.hashes:
                if hamming(h, other) <= thresh:
                    return None
            self.hashes.append(h)
            self.idx += 1
            return self.idx


def work(cand, args, store, out_dir, label):
    if cand["source"] == "wikimedia_commons":
        url, ua = commons_thumb(cand["url"], args.long_side * 2), UA_COM
    else:
        url = re.sub(r"/(square|small|medium|large|original|thumbnail)\.",
                     f"/{args.inat_size}.", cand["url"])
        ua = UA_INAT
    raw = fetch(url, ua)
    if raw is None:
        return cand, "download_failed", None
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            if im.mode != "RGB":
                im = im.convert("RGB")
            w, h = im.size
            if max(w, h) < args.min_side:
                return cand, f"too_small_{w}x{h}", None
            if max(w / h, h / w) > 3.0:
                return cand, "extreme_aspect", None
            if max(w, h) > args.long_side:
                im.thumbnail((args.long_side, args.long_side), Image.LANCZOS)
            if args.sharpen:
                from PIL import ImageFilter
                im = im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=55, threshold=3))
            n = store.claim(dhash(im), args.dup_thresh)
            if n is None:
                return cand, "near_duplicate", None
            safe = re.sub(r"[^A-Za-z0-9._-]", "", cand["uid"])[-40:]
            name = f"{label}__{n:05d}_{cand['source'][:4]}_{safe}.jpg"
            fp = os.path.join(out_dir, name)
            im.save(fp, "JPEG", quality=args.quality, optimize=True, progressive=True)
            rec = {
                "file": f"{label}/{name}", "label": label,
                "class_index": LABELS.index(label), "zh_name": META[label]["zh"],
                "sci_name": META[label]["sci"], "source": cand["source"],
                "image_id": cand["uid"], "w": im.size[0], "h": im.size[1],
                "orig_w": cand.get("orig_w"), "orig_h": cand.get("orig_h"),
                "bytes": os.path.getsize(fp), "license": cand.get("license"),
                "commercial_ok": classify_license(cand.get("license")),
                "attribution": cand.get("attribution"), "grade": cand.get("grade"),
                "faves": cand.get("faves"), "latlng": cand.get("latlng"),
                "observed_on": cand.get("observed_on"),
                "page_url": cand.get("obs_uri") or cand.get("page") or url,
                "image_url": url,
            }
            return cand, None, rec
    except Exception as e:
        return cand, f"error_{type(e).__name__}", None


def existing_hashes(out_dir, label, args):
    d = os.path.join(out_dir, label)
    hashes, n_max = [], 0
    if not os.path.isdir(d) or args.fresh:
        return hashes, 0
    for f in sorted(os.listdir(d)):
        if not f.endswith(".jpg"):
            continue
        m = re.match(r".*?__(\d{5})_", f)
        if m:
            n_max = max(n_max, int(m.group(1)))
        try:
            with Image.open(os.path.join(d, f)) as im:
                im.load()
                hashes.append(dhash(im))
        except Exception:
            pass
    return hashes, n_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label", type=int, default=30, help="final images kept per species")
    ap.add_argument("--long-side", type=int, default=1024, help="max long side in px")
    ap.add_argument("--min-side", type=int, default=700, help="reject if long side below this")
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--dup-thresh", type=int, default=6, help="dHash Hamming radius")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sharpen", action="store_true")
    ap.add_argument("--inat-size", default="large", choices=["medium", "large", "original"])
    ap.add_argument("--commons-ratio", type=float, default=0.15)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "images"))
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    bl = os.path.join(ROOT, "blocklist.txt")
    blocked = {x.strip() for x in open(bl, encoding="utf-8")} if os.path.exists(bl) else set()
    if blocked:
        print(f"blocklist: {len(blocked)} uids excluded")
    want = [l for l in LABELS if not args.only or l in args.only.split(",")]
    all_rows, all_rej = [], []
    for label in want:
        d = os.path.join(args.out, label)
        os.makedirs(d, exist_ok=True)
        cands = [json.loads(x) for x in open(os.path.join(ROOT, "candidates", f"{label}.jsonl"),
                                             encoding="utf-8")]
        inat = [c for c in cands if c["source"] == "inaturalist" and c["uid"] not in blocked]
        com = [c for c in cands if c["source"] != "inaturalist" and c["uid"] not in blocked]
        hashes, n_max = existing_hashes(args.out, label, args)
        have = len(hashes)
        need = max(0, args.per_label - have)
        print(f"[{label:22s}] have={have:<4} target={args.per_label} "
              f"pool(inat={len(inat)},commons={len(com)})", flush=True)
        if need == 0:
            continue
        n_com = int(round(args.per_label * args.commons_ratio))
        plan = com[:n_com] + inat + com[n_com:]
        store = Store(hashes, n_max)
        rows, rej = [], []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, c, args, store, d, label)
                    for c in plan[:min(len(plan), need + max(30, need // 3))]]
            for fu in as_completed(futs):
                cand, why, rec = fu.result()
                if rec:
                    rows.append(rec)
                else:
                    rej.append({"label": label, "uid": cand["uid"], "reason": why,
                                "url": cand["url"]})
                if len(rows) >= need:
                    break
            for fu in futs:
                fu.cancel()
        print(f"    kept={len(rows)} rejected={len(rej)}", flush=True)
        all_rows += rows
        all_rej += rej

    cols = ["file", "label", "class_index", "zh_name", "sci_name", "source", "image_id",
            "w", "h", "orig_w", "orig_h", "bytes", "license", "commercial_ok", "attribution",
            "grade", "faves", "latlng", "observed_on", "page_url", "image_url"]
    # merge with the manifest already on disk so incremental runs stay complete
    jp = os.path.join(ROOT, "manifest.jsonl")
    cp = os.path.join(ROOT, "manifest.csv")
    prev = []
    if os.path.exists(jp):
        prev = [json.loads(x) for x in open(jp, encoding="utf-8")]
    seen = {r["file"] for r in prev}
    merged = prev + [r for r in all_rows if r["file"] not in seen]
    merged.sort(key=lambda r: (r["class_index"], r["file"]))
    with open(cp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)
    with open(jp, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(ROOT, "rejected.jsonl"), "w", encoding="utf-8") as f:
        for r in all_rej:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"classes": LABELS, "class_map": {l: i for i, l in enumerate(LABELS)},
               "zh": {l: META[l]["zh"] for l in LABELS},
               "sci": {l: META[l]["sci"] for l in LABELS}},
              open(os.path.join(ROOT, "classes.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\nmanifest rows: {len(merged)}")


if __name__ == "__main__":
    main()
