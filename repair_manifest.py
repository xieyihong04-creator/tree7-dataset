#!/usr/bin/env python3
"""Rebuild manifest.csv/jsonl from what is actually on disk, then audit it.

The builder can leave a few images written but unlisted (tasks that finished
after the count target was hit), so the filesystem is treated as the source of
truth and metadata is re-joined from candidates/*.jsonl by image id.
"""
import csv, glob, json, os, re, collections
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
IMGS = os.path.join(ROOT, "images")
CFG = json.load(open(os.path.join(ROOT, "species.json"), encoding="utf-8"))
LABELS = [s["label"] for s in CFG["species"]]
META = {s["label"]: s for s in CFG["species"]}

NC_KEYS = ("cc-by-nc", "cc-nc", "by-nc")


def classify_license(code):
    c = (code or "").lower().replace(" ", "-")
    if not c:
        return "unknown"
    if any(k in c for k in NC_KEYS):
        return "no"
    if c in ("cc-by", "cc-by-sa", "cc-by-nd", "cc0", "cc-pd", "pd", "public-domain") \
       or re.match(r"^cc-by(-sa)?-[0-9.]+$", c) or "public domain" in c or c.startswith("gfdl"):
        return "yes"
    return "unknown"


def dhash(img, size=8):
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = np.asarray(g, dtype=np.int16)
    return np.packbits((px[:, :-1] < px[:, 1:]).ravel()).reshape(-1)


cand, by_tail = {}, {}
for f in glob.glob(os.path.join(ROOT, "candidates", "*.jsonl")):
    for line in open(f, encoding="utf-8"):
        c = json.loads(line)
        cand.setdefault(c["uid"], c)
        # build.py sanitises then truncates the uid when naming files
        tail = re.sub(r"[^A-Za-z0-9._-]", "", c["uid"])[-40:]
        by_tail.setdefault(tail, c)

rows, missing, hashes = [], [], []
for label in LABELS:
    d = os.path.join(IMGS, label)
    for path in sorted(glob.glob(os.path.join(d, "*.jpg"))):
        name = os.path.basename(path)
        m = re.match(rf"{re.escape(label)}__\d{{5}}_(\w\w\w\w)_(.+)\.jpg$", name)
        uid = m.group(2) if m else None
        c = cand.get(uid) or by_tail.get(uid)
        if c is None:
            missing.append(name)
            continue
        with Image.open(path) as im:
            w, h = im.size
            hashes.append((label, name, dhash(im)))
        rows.append({
            "file": f"{label}/{name}", "label": label, "class_index": LABELS.index(label),
            "zh_name": META[label]["zh"], "sci_name": META[label]["sci"],
            "source": c["source"], "image_id": uid, "w": w, "h": h,
            "orig_w": c.get("orig_w"), "orig_h": c.get("orig_h"),
            "bytes": os.path.getsize(path), "license": c.get("license"),
            "commercial_ok": classify_license(c.get("license")),
            "attribution": c.get("attribution"), "grade": c.get("grade"),
            "faves": c.get("faves"), "latlng": c.get("latlng"),
            "observed_on": c.get("observed_on"),
            "page_url": c.get("obs_uri") or c.get("page") or c.get("url"),
            "image_url": c.get("url"),
        })

cols = ["file", "label", "class_index", "zh_name", "sci_name", "source", "image_id",
        "w", "h", "orig_w", "orig_h", "bytes", "license", "commercial_ok", "attribution",
        "grade", "faves", "latlng", "observed_on", "page_url", "image_url"]
rows.sort(key=lambda r: (r["class_index"], r["file"]))
with open(os.path.join(ROOT, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
with open(os.path.join(ROOT, "manifest.jsonl"), "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---- audit -----------------------------------------------------------------
print(f"images listed: {len(rows)}   unmatched files: {len(missing)}")
if missing:
    print("  e.g.", missing[:5])
by_lab = collections.Counter(r["label"] for r in rows)
for l in LABELS:
    print(f"  {l:22s} {by_lab[l]:>4}")
print("source:", collections.Counter(r["source"] for r in rows).most_common())
print("license:", collections.Counter(str(r["license"]).lower() for r in rows).most_common(6))
print("commercial_ok:", collections.Counter(r["commercial_ok"] for r in rows).most_common())
print("grade:", collections.Counter(str(r["grade"]) for r in rows).most_common())
mb = sum(r["bytes"] for r in rows) / 1e6
print(f"total {mb:.0f} MB, mean {mb*1e6/len(rows)/1024:.0f} KB/img")
res = np.array([[r["w"], r["h"]] for r in rows])
print(f"resolution: p5={np.percentile(res.max(1),5):.0f} p50={np.percentile(res.max(1),50):.0f} "
      f"min_side_p5={np.percentile(res.min(1),5):.0f}")

# cross-class near-duplicates = possible label noise
H = np.stack([h for _, _, h in hashes])
lab = np.array([l for l, _, _ in hashes])
dupes = []
for i in range(len(H)):
    diff = np.unpackbits(H ^ H[i], axis=1).sum(1)
    same = diff <= 6
    same[i] = False
    cross = same & (lab != lab[i])
    if cross.any():
        dupes.append((lab[i], hashes[i][1], int(diff[cross].min())))
print("cross-class near-duplicates:", len(dupes) // 2)
for d in dupes[:8]:
    print("  ", d)
