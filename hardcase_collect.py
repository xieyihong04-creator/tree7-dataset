#!/usr/bin/env python3
"""Collect a whole-tree/canopy hard-case pool for the Ficus/Terminalia pair.

Pulls research-grade iNaturalist photos (metadata only kept in the repo; image
files stay out of git) and renders numbered contact sheets so a human can pick
the whole-tree shots.  Selection is written to hardcases/selection.json.

Usage:
    python3 hardcase_collect.py --per-class 150        # download + sheets
    python3 hardcase_collect.py --sheets-only          # rebuild sheets
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
import time

import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
HC = os.path.join(ROOT, "hardcases")
RAW = os.path.join(HC, "raw")
SHEETS = os.path.join(HC, "sheets")
META = os.path.join(HC, "metadata.jsonl")
UA = {"User-Agent": "tree7-hardcase-builder/0.1 (research dataset; contact via iNaturalist)"}
INAT = "https://api.inaturalist.org/v1"

TAXA = {
    "Ficus_microcarpa": 54982,
    "Terminalia_catappa": 62821,
}
ZH = {"Ficus_microcarpa": "榕樹", "Terminalia_catappa": "大葉欖仁"}
THUMB = 320
COLS, ROWS = 5, 4


def get(url, **kw):
    for attempt in range(5):
        try:
            r = requests.get(url, headers=UA, timeout=60, **kw)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r
        if r.status_code in (403, 429, 500, 502, 503):
            time.sleep(min(60, 4 * (attempt + 1)))
            continue
        print(f"  HTTP {r.status_code} {url}", file=sys.stderr)
        return None
    return None


def large_url(photo_url: str) -> str:
    return re.sub(r"/(square|small|medium|large|original)\.(jpe?g|png)$", r"/large.\2", photo_url)


def collect(label: str, taxon_id: int, per_class: int) -> list[dict]:
    out, seen = [], set()
    page = 1
    while len(out) < per_class and page <= 12:
        url = f"{INAT}/observations"
        params = {
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "photos": "true",
            "per_page": 50,
            "page": page,
            "order_by": "faves_count" if page <= 3 else "created_at",
            "sort": "desc",
        }
        r = get(url, params=params)
        if r is None:
            break
        results = r.json().get("results", [])
        if not results:
            break
        for obs in results:
            photos = obs.get("photos") or []
            if not photos:
                continue
            ph = photos[0]
            pid = ph["id"]
            if pid in seen:
                continue
            seen.add(pid)
            dims = ph.get("original_dimensions") or {}
            if min(dims.get("width", 0), dims.get("height", 0)) < 700:
                continue
            out.append(
                {
                    "class": label,
                    "photo_id": pid,
                    "obs_id": obs["id"],
                    "page_url": f"https://www.inaturalist.org/observations/{obs['id']}",
                    "image_url": large_url(ph["url"]),
                    "license": ph.get("license_code"),
                    "attribution": ph.get("attribution"),
                    "observed_on": obs.get("observed_on"),
                    "lat": obs.get("geojson", {}).get("coordinates", [None, None])[1],
                    "lng": obs.get("geojson", {}).get("coordinates", [None, None])[0],
                    "faves": obs.get("faves_count", 0),
                }
            )
            if len(out) >= per_class:
                break
        page += 1
        time.sleep(0.8)
    return out


def download(records: list[dict]) -> None:
    for i, rec in enumerate(records):
        folder = os.path.join(RAW, rec["class"])
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{rec['photo_id']}.jpg")
        rec["file"] = os.path.relpath(path, ROOT)
        if os.path.exists(path):
            continue
        r = get(rec["image_url"])
        if r is None:
            print(f"  skip {rec['photo_id']}", file=sys.stderr)
            continue
        try:
            with Image.open(io.BytesIO(r.content)) as im:
                im.convert("RGB")
        except Exception:
            print(f"  broken {rec['photo_id']}", file=sys.stderr)
            continue
        with open(path, "wb") as fh:
            fh.write(r.content)
        if (i + 1) % 25 == 0:
            print(f"  {rec['class']}: {i + 1}/{len(records)}", flush=True)
        time.sleep(0.3)


def font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_sheets(records: list[dict]) -> None:
    os.makedirs(SHEETS, exist_ok=True)
    fnt = font(28)
    fnt_small = font(20)
    by_class = {}
    for rec in records:
        by_class.setdefault(rec["class"], []).append(rec)
    for label, recs in by_class.items():
        recs.sort(key=lambda r: r["photo_id"])
        per_sheet = COLS * ROWS
        for s in range(math.ceil(len(recs) / per_sheet)):
            chunk = recs[s * per_sheet:(s + 1) * per_sheet]
            sheet = Image.new("RGB", (COLS * THUMB, ROWS * THUMB), "white")
            draw = ImageDraw.Draw(sheet)
            for i, rec in enumerate(chunk):
                x, y = (i % COLS) * THUMB, (i // COLS) * THUMB
                try:
                    with Image.open(os.path.join(ROOT, rec["file"])) as im:
                        im = im.convert("RGB")
                        im.thumbnail((THUMB - 8, THUMB - 8))
                        ox = x + (THUMB - im.width) // 2
                        oy = y + (THUMB - im.height) // 2
                        sheet.paste(im, (ox, oy))
                except Exception:
                    pass
                idx = s * per_sheet + i
                text = f"{label.split('_')[0][:4]} {idx:03d}"
                draw.rectangle([x, y, x + 128, y + 30], fill="black")
                draw.text((x + 5, y + 2), text, fill="yellow", font=fnt)
                draw.text((x + 5, y + THUMB - 26), f"fav{rec['faves']}", fill="red", font=fnt_small)
            path = os.path.join(SHEETS, f"{label}_{s:02d}.jpg")
            sheet.save(path, quality=88)
            print(f"  sheet {os.path.relpath(path, ROOT)} ({len(chunk)} imgs)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--sheets-only", action="store_true")
    ap.add_argument("--classes", default="Ficus_microcarpa,Terminalia_catappa")
    args = ap.parse_args()

    os.makedirs(HC, exist_ok=True)
    labels = [c.strip() for c in args.classes.split(",") if c.strip()]

    if args.sheets_only and os.path.exists(META):
        records = [json.loads(line) for line in open(META, encoding="utf-8")]
    else:
        records = []
        for label in labels:
            print(f"collecting {label} …", flush=True)
            recs = collect(label, TAXA[label], args.per_class)
            print(f"  {len(recs)} candidates")
            download(recs)
            records += [r for r in recs if os.path.exists(os.path.join(ROOT, r["file"]))]
        with open(META, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    build_sheets(records)
    print(f"total candidates: {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
