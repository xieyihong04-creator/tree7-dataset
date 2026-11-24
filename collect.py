#!/usr/bin/env python3
"""Stage 1: build a candidate index (metadata + URLs, no image bytes yet).

Sources: iNaturalist open API (research-grade IDs first) and Wikimedia Commons
category files.  Candidates are pre-filtered by the resolution reported in the
API metadata so we never download obviously useless thumbnails.
"""
import argparse, json, os, re, sys, time, urllib.parse
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(ROOT, "species.json"), encoding="utf-8"))
TAXA_CACHE = os.path.join(ROOT, "taxa.json")
CAND_DIR = os.path.join(ROOT, "candidates")
UA = {"User-Agent": "tree-dataset-builder/0.1 (research dataset; contact via iNaturalist app)"}
# Wikimedia's edge rejects generic/bot-looking agent strings with 403; it wants
# "tool/version (url; contact) language-runtime" format.
UA_COM = {"User-Agent": "tree-dataset-builder/0.1 (https://qoder.com; research image dataset) python-requests"}

INAT = "https://api.inaturalist.org/v1"
GX = "https://inaturalist-open-data.s3.amazonaws.com/photos"


class Throttle:
    """Simple per-host rate limiter with backoff on 403/429/5xx."""

    def __init__(self, min_interval):
        self.min = min_interval
        self.last = 0.0

    def wait(self):
        dt = time.time() - self.last
        if dt < self.min:
            time.sleep(self.min - dt)
        self.last = time.time()


inat_rl = Throttle(1.2)
gx_rl = Throttle(0.0)
com_rl = Throttle(1.0)


def req_json(url, rl, params=None, tries=5, headers=None):
    hdr = headers or UA
    for attempt in range(tries):
        rl.wait()
        try:
            r = requests.get(url, params=params, headers=hdr, timeout=60)
        except requests.RequestException as e:
            time.sleep(min(60, 2 ** attempt * 3))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (403, 429, 500, 502, 503):
            ra = r.headers.get("Retry-After")
            sleep_s = float(ra) if ra and ra.isdigit() else min(90, 5 * (attempt + 1))
            print(f"    retry {r.status_code} in {sleep_s:.0f}s", file=sys.stderr)
            time.sleep(sleep_s)
            continue
        print(f"    HTTP {r.status_code} {url}", file=sys.stderr)
        return None
    return None


# ---------------------------------------------------------------- taxa ids
def load_taxa(force=False):
    if os.path.exists(TAXA_CACHE) and not force:
        return json.load(open(TAXA_CACHE, encoding="utf-8"))
    taxa = {}
    for sp in CFG["species"]:
        sci = sp["sci"]
        d = req_json(f"{INAT}/taxa", inat_rl,
                     params={"q": sci, "rank": "species", "limit": 5, "inat_id": True})
        hit = None
        for r in (d or {}).get("results", []):
            if r["name"].replace(" ", "_") == sci.replace(" ", "_") and r.get("is_active", True):
                hit = r
                break
        if hit is None and (d or {}).get("results"):
            hit = d["results"][0]
        if hit is None:
            print(f"  !! no iNaturalist taxon for {sci}", file=sys.stderr)
            continue
        obs = req_json(f"{INAT}/observations", inat_rl,
                       params={"taxon_id": hit["id"], "has_photos": "true", "per_page": 0})
        res = req_json(f"{INAT}/observations", inat_rl,
                       params={"taxon_id": hit["id"], "has_photos": "true",
                               "quality_grade": "research", "per_page": 0})
        taxa[sp["label"]] = {
            "taxon_id": hit["id"], "name": hit["name"], "rank": hit.get("rank"),
            "photo_obs": (obs or {}).get("total_results", 0),
            "research_obs": (res or {}).get("total_results", 0),
        }
        print(f"  {sci:22s} iNat#{hit['id']:<7} photo_obs={taxa[sp['label']]['photo_obs']:>7} "
              f"research={taxa[sp['label']]['research_obs']:>7}")
    json.dump(taxa, open(TAXA_CACHE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return taxa


# ---------------------------------------------------------------- inaturalist
def inat_url(photo_url, size):
    return re.sub(r"/(square|small|medium|large|original)\.(jpe?g|png|gif|tiff)$",
                  f"/{size}.\\2", photo_url)


def collect_inat(label, taxon_id, per_source, use_needs_id=False):
    """Pull photo records; prefer research grade, optionally widen the pool."""
    seen_obs, out = set(), []
    passes = [{"quality_grade": "research", "order_by": "favorites_count", "sort": "desc"},
              {"quality_grade": "research", "order_by": "created_at", "sort": "desc"}]
    if use_needs_id:
        # research grade first, then progressively weaker labels
        passes = [{"quality_grade": "research", "order_by": "favorites_count", "sort": "desc"},
                  {"quality_grade": "research", "order_by": "created_at", "sort": "desc"},
                  {"quality_grade": "needs_id", "order_by": "favorites_count", "sort": "desc"},
                  {"order_by": "favorites_count", "sort": "desc"}]
    for p in passes:
        page = 1
        while len(out) < per_source and page <= 10:  # iNat caps page*per_page at 1000
            params = {"taxon_id": taxon_id, "has_photos": "true", "per_page": 100,
                      "page": page, "fields": ",".join([
                          "id", "quality_grade", "license_code", "faves_count", "location",
                          "observed_on", "uri", "photos.id", "photos.license_code",
                          "photos.attribution", "photos.url", "photos.original_dimensions"])}
            params.update(p)
            d = req_json(f"{INAT}/observations", inat_rl, params=params)
            if not d or not d.get("results"):
                break
            for o in d["results"]:
                if o["id"] in seen_obs:
                    continue
                seen_obs.add(o["id"])
                for ph in (o.get("photos") or [])[:2]:
                    od = ph.get("original_dimensions") or {}
                    w, h = od.get("width") or 0, od.get("height") or 0
                    if min(w, h) < 900:          # skip phone-cropped junk / stamps
                        continue
                    out.append({
                        "source": "inaturalist",
                        "uid": f"inat_{ph['id']}",
                        "url": inat_url(ph["url"], "large"),
                        "orig_w": w, "orig_h": h,
                        "license": ph.get("license_code") or "CC-BY-NC",
                        "attribution": ph.get("attribution"),
                        "obs_id": o["id"], "obs_uri": o.get("uri"),
                        "grade": o.get("quality_grade"), "faves": o.get("faves_count", 0),
                        "latlng": o.get("location"), "observed_on": o.get("observed_on"),
                    })
            if len(out) >= per_source:
                break
            page += 1
        time.sleep(0.5)
    return out


# ---------------------------------------------------------------- commons
ART_RE = re.compile(r"<[^>]+>|&#160;|&quot;|&amp;")


def clean_html(s):
    if not s:
        return ""
    s = ART_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


BINOMIAL = re.compile(r"([A-Z][a-z]{3,})_[a-z]{3,}")
GENERA = {x["label"]: {x["sci"].split()[0].lower(),
                       *(g.lower() for g in x.get("syn_genera", []))}
          for x in CFG["species"]}


def wrong_genus(title, allowed):
    """Reject Commons files whose title names a different genus (mislabeled uploads)."""
    if not allowed:
        return False
    for g in BINOMIAL.findall(title):
        if g.lower() not in allowed:
            return True
    return False


def collect_commons(label, cats, per_source):
    allowed = GENERA.get(label, set())
    out, names = [], set()
    for cat in cats:
        cont = None
        while len(out) < per_source:
            params = {"action": "query", "generator": "categorymembers",
                      "gcmtitle": f"Category:{cat}", "gcmtype": "file", "gcmlimit": 200,
                      "prop": "imageinfo",
                      "iiprop": "url|size|mime|extmetadata|timestamp",
                      "iiextmetadatafilter": "License|LicenseShortName|Artist|UsageTerms|LicenseURL",
                      "format": "json"}
            if cont:
                params.update(cont)
            d = req_json("https://commons.wikimedia.org/w/api.php", com_rl, params=params,
                         headers=UA_COM)
            if not d or "query" not in d:
                break
            for pg in d["query"].get("pages", {}).values():
                title = pg.get("title", "")
                if title in names:
                    continue
                ii = (pg.get("imageinfo") or [{}])[0]
                mime = (ii.get("mime") or "").lower()
                if mime not in ("image/jpeg", "image/png"):
                    continue
                if min(ii.get("width") or 0, ii.get("height") or 0) < 900:
                    continue
                if re.search(r"(map|diagram|illustration|logo|herbarium|scan|drawing|"
                             r"line drawing|distribution|chart|bhl|page \d+|plate \d+|"
                             r"poster|banner|couplet|chs-\d+|misidentified)", title, re.I):
                    continue
                if wrong_genus(title, allowed):
                    continue
                em = ii.get("extmetadata", {})
                names.add(title)
                out.append({
                    "source": "wikimedia_commons",
                    "uid": "wc_" + re.sub(r"\W+", "_", title)[:90],
                    "url": ii.get("url"), "thumb_fallback": True,
                    "orig_w": ii["width"], "orig_h": ii["height"],
                    "license": (em.get("LicenseShortName", {}) or {}).get("value", "unknown"),
                    "license_url": (em.get("LicenseURL", {}) or {}).get("value", ""),
                    "attribution": clean_html((em.get("Artist", {}) or {}).get("value", "")),
                    "obs_uri": "https://commons.wikimedia.org/wiki/" + title.replace(" ", "_"),
                    "observed_on": (em.get("DateTime", {}) or {}).get("value", "")[:10],
                    "page": ii.get("descriptionurl"),
                })
                if len(out) >= per_source:
                    break
            cont = d.get("continue")
            if not cont:
                break
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=400,
                    help="max candidates collected per source, per species")
    ap.add_argument("--only", default="", help="comma list of labels")
    ap.add_argument("--refresh-taxa", action="store_true")
    ap.add_argument("--wide", action="store_true",
                    help="also include needs_id observations (small-pool species)")
    args = ap.parse_args()

    os.makedirs(CAND_DIR, exist_ok=True)
    print("[1/2] resolving iNaturalist taxa")
    taxa = load_taxa(args.refresh_taxa)

    print("[2/2] collecting candidates")
    want = [s for s in CFG["species"] if not args.only or s["label"] in args.only.split(",")]
    for sp in want:
        label = sp["label"]
        cands, stats = [], []
        t = taxa.get(label)
        if t:
            small = t["research_obs"] < args.per_source
            got = collect_inat(label, t["taxon_id"], args.per_source,
                               use_needs_id=args.wide or small)
            cands += got
            stats.append(f"inaturalist={len(got)}")
        else:
            stats.append("inaturalist=0(no taxon)")
        got = collect_commons(label, CFG["commons_categories"].get(label, [sp["sci"].replace(" ", "_")]),
                              max(60, args.per_source // 3))
        cands += got
        stats.append(f"commons={len(got)}")

        dedup, seen = [], set()
        for c in cands:
            if c["uid"] in seen:
                continue
            seen.add(c["uid"])
            c["label"] = label
            c["sci"] = sp["sci"]
            c["zh"] = sp["zh"]
            dedup.append(c)
        path = os.path.join(CAND_DIR, f"{label}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for c in dedup:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"  {label:22s} {len(dedup):>5} candidates  ({', '.join(stats)})")
    print("done ->", CAND_DIR)


if __name__ == "__main__":
    main()
