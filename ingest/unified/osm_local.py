"""OpenStreetMap from disk: the roads, buildings and car parks around a point in milliseconds.

The public Overpass server can take half a minute to answer anything. Every property in the
kit is in greater Los Angeles or greater Houston, so the two regional extracts are indexed
once and read locally; Overpass is only asked for points outside them.

    uv run python -m ingest.unified.osm_local --unpack        # use the site pack (shared privately; gitignored)
    uv run python -m ingest.unified.osm_local 34.0323 -118.4522   # what is around a point

    uv run python -m ingest.unified.osm_local --download      # fetch the two extracts (about 1.4 GB)
    uv run python -m ingest.unified.osm_local                 # build the full index (about 3 GB, 15 min)
    uv run python -m ingest.unified.osm_local --pack          # cut the site pack out of the full index

Two indexes, one shape
----------------------
* The **full index** covers the whole Los Angeles and Houston boxes, so any address in them is
  answered locally. It is built from the regional extracts and is far too large to commit.
* The **site pack** (``data/osm/site_pack.sqlite.gz``, gitignored, shared privately) holds only what lies within
  ``PACK_REACH_M`` of a property in the kit. It answers for those properties and nothing else,
  and is unpacked to the index path on first use when no index is there. Nobody who only
  works with the kit's properties needs to download or build anything.

What is kept (inside the regions' boxes only):

* ways tagged ``highway`` with one of Track A's road classes (centrelines);
* areas (closed ways and multipolygon relations) tagged ``building``, ``amenity=parking``
  or ``landuse=residential``, one element per outer ring;
* named nodes tagged ``amenity``, ``shop``, ``tourism`` or ``leisure`` (places a caller
  might name).

``fetch_site`` returns the same shape as an Overpass ``out geom tags`` answer
(``{"elements": [...]}``), so everything downstream reads either source unchanged.
The index is a cache of public map data (ODbL, (c) OpenStreetMap contributors): it lives
under ``data/`` and is not committed.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACT_DIR = REPO_ROOT / "data" / "raw" / "osm"
INDEX_PATH = REPO_ROOT / "data" / "processed" / "osm_index.sqlite"
PACK_PATH = REPO_ROOT / "data" / "osm" / "site_pack.sqlite.gz"
PROPERTY_INDEX = REPO_ROOT / "data" / "processed" / "property_index.sqlite"
INDEX_VERSION = 1
PACK_REACH_M = 230.0   # the pack keeps everything this close to a property
SITE_MATCH_M = 60.0    # a lookup this close to a packed property is answered from the pack

# name -> (extract file, download URL, (south, west, north, east) of what is kept)
REGIONS = {
    "los-angeles": ("socal.osm.pbf", "https://download.geofabrik.de/north-america/us/california/socal-latest.osm.pbf",
                    (33.60, -118.75, 34.45, -117.85)),
    "houston": ("texas.osm.pbf", "https://download.geofabrik.de/north-america/us/texas-latest.osm.pbf",
                (29.30, -95.95, 30.25, -94.85)),
}
KEPT_TAGS = ("name", "highway", "service", "building", "amenity", "parking", "landuse", "shop", "tourism", "leisure")
PLACE_KEYS = ("amenity", "shop", "tourism", "leisure")
M_PER_DEG_LAT = 111_320.0


def _tags(obj) -> dict:
    return {k: obj.tags[k] for k in KEPT_TAGS if k in obj.tags}


def _ring(nodes) -> list[dict]:
    return [{"lat": round(n.lat, 7), "lon": round(n.lon, 7)} for n in nodes if n.location.valid()]


def _elements(pbf: Path, box: tuple[float, float, float, float]):
    """Overpass-shaped elements of one extract that touch ``box``."""
    import osmium
    from ingest.mp3d.aerial import ROAD_CLASS_RANK

    south, west, north, east = box

    def touches(geometry: list[dict]) -> bool:
        return any(south <= p["lat"] <= north and west <= p["lon"] <= east for p in geometry)

    keys = osmium.filter.KeyFilter("highway", "building", "amenity", "landuse", "shop", "tourism", "leisure")
    for o in osmium.FileProcessor(str(pbf)).with_locations().with_areas().with_filter(keys):
        tags = o.tags
        if o.is_node():
            if "name" in tags and any(k in tags for k in PLACE_KEYS) and south <= o.lat <= north and west <= o.lon <= east:
                yield {"type": "node", "id": o.id, "tags": _tags(o), "lat": round(o.lat, 7), "lon": round(o.lon, 7)}
        elif o.is_way():
            if tags.get("highway") in ROAD_CLASS_RANK:
                geometry = _ring(o.nodes)
                if len(geometry) >= 2 and touches(geometry):
                    yield {"type": "way", "id": o.id, "tags": _tags(o), "geometry": geometry}
        elif o.is_area():
            if not ("building" in tags or tags.get("amenity") == "parking" or tags.get("landuse") == "residential"):
                continue
            for i, outer in enumerate(o.outer_rings()):
                geometry = _ring(outer)
                if len(geometry) >= 4 and touches(geometry):
                    # one element per outer ring, shaped like a closed way; an area's id is unique
                    # across ways and relations, and the ring number keeps a multipolygon's parts apart
                    yield {"type": "way", "id": o.id * 100 + i, "tags": _tags(o), "geometry": geometry}


def download(force: bool = False) -> None:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (filename, url, _box) in REGIONS.items():
        path = EXTRACT_DIR / filename
        if path.is_file() and not force:
            print(f"{name}: {path.name} already on disk ({path.stat().st_size / 1e6:.0f} MB)")
            continue
        print(f"{name}: downloading {url}")
        urllib.request.urlretrieve(url, path)
        print(f"{name}: {path.stat().st_size / 1e6:.0f} MB")


def build_index(index_path: Optional[Path] = None, extract_dir: Optional[Path] = None, regions: Optional[dict] = None) -> dict:
    """Read the extracts and (re)write the index. Returns ``{region: element count}``."""
    index_path, extract_dir = index_path or INDEX_PATH, extract_dir or EXTRACT_DIR
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    db.executescript(
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);"
        "CREATE TABLE elements (rid INTEGER PRIMARY KEY, body TEXT);"
        "CREATE VIRTUAL TABLE boxes USING rtree(rid, south, north, west, east);")
    counts: dict[str, int] = {}
    covered: dict[str, list[float]] = {}
    rid = 0
    for name, (filename, _url, box) in (regions or REGIONS).items():
        pbf = extract_dir / filename
        if not pbf.is_file():
            print(f"{name}: {pbf} is missing (run with --download); skipped", file=sys.stderr)
            continue
        started, n = time.perf_counter(), 0
        for element in _elements(pbf, box):
            points = element.get("geometry") or [element]
            lats, lons = [p["lat"] for p in points], [p["lon"] for p in points]
            rid += 1
            n += 1
            db.execute("INSERT INTO elements VALUES (?, ?)", (rid, json.dumps(element, separators=(",", ":"))))
            db.execute("INSERT INTO boxes VALUES (?, ?, ?, ?, ?)", (rid, min(lats), max(lats), min(lons), max(lons)))
        counts[name], covered[name] = n, list(box)
        print(f"{name}: {n:,} elements in {time.perf_counter() - started:.0f} s")
    db.execute("INSERT INTO meta VALUES ('version', ?)", (str(INDEX_VERSION),))
    db.execute("INSERT INTO meta VALUES ('regions', ?)", (json.dumps(covered),))
    db.execute("INSERT INTO meta VALUES ('built', ?)", (time.strftime("%Y-%m-%d"),))
    db.commit()
    db.close()
    _close_all()
    tmp.replace(index_path)
    return counts


def _reach(lat: float, radius_m: float) -> tuple[float, float]:
    return radius_m / M_PER_DEG_LAT, radius_m / (M_PER_DEG_LAT * max(math.cos(math.radians(lat)), 0.01))


def property_points(property_index: Optional[Path] = None) -> list[tuple[float, float]]:
    """(lat, lng) of every property in the kit's address index."""
    db = sqlite3.connect(property_index or PROPERTY_INDEX)
    try:
        return [(float(a), float(b)) for a, b in
                db.execute("SELECT lat, lng FROM properties WHERE lat IS NOT NULL AND lng IS NOT NULL ORDER BY id")]
    finally:
        db.close()


def export_pack(points: list[tuple[float, float]], pack_path: Optional[Path] = None,
                index_path: Optional[Path] = None, reach_m: float = PACK_REACH_M) -> dict:
    """Cut a small index out of the full one: everything within ``reach_m`` of the given points, gzipped."""
    import gzip
    import shutil

    opened = _open(index_path)
    if not opened or opened[1].get("coverage") == "sites":
        raise RuntimeError("the site pack is cut from the full index: build that first (see the module docstring)")
    src = opened[0]
    pack_path = pack_path or PACK_PATH
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pack_path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    out = sqlite3.connect(tmp)
    out.executescript(
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);"
        "CREATE TABLE elements (rid INTEGER PRIMARY KEY, body TEXT);"
        "CREATE VIRTUAL TABLE boxes USING rtree(rid, south, north, west, east);"
        "CREATE VIRTUAL TABLE sites USING rtree(sid, south, north, west, east);")
    kept: set[int] = set()
    sites = 0
    for sid, (lat, lon) in enumerate(points, start=1):
        if not _in_regions(opened[1], lat, lon, reach_m):
            continue
        dlat, dlon = _reach(lat, reach_m)
        rows = src.execute(
            "SELECT b.rid, b.south, b.north, b.west, b.east, e.body FROM boxes b JOIN elements e ON e.rid = b.rid "
            "WHERE b.north >= ? AND b.south <= ? AND b.east >= ? AND b.west <= ?",
            (lat - dlat, lat + dlat, lon - dlon, lon + dlon)).fetchall()
        for rid, s, n, w, e, body in rows:
            if rid not in kept:
                kept.add(rid)
                out.execute("INSERT INTO elements VALUES (?, ?)", (rid, body))
                out.execute("INSERT INTO boxes VALUES (?, ?, ?, ?, ?)", (rid, s, n, w, e))
        out.execute("INSERT INTO sites VALUES (?, ?, ?, ?, ?)", (sid, lat, lat, lon, lon))
        sites += 1
    for key, value in (("version", str(INDEX_VERSION)), ("coverage", "sites"), ("reach_m", str(reach_m)),
                       ("built", opened[1].get("built") or ""), ("regions", "{}")):
        out.execute("INSERT INTO meta VALUES (?, ?)", (key, value))
    out.commit()
    out.execute("VACUUM")
    out.close()
    with open(tmp, "rb") as fh, gzip.GzipFile(pack_path, "wb", compresslevel=9, mtime=0) as gz:
        shutil.copyfileobj(fh, gz, 1 << 20)
    raw = tmp.stat().st_size
    tmp.unlink()
    return {"sites": sites, "elements": len(kept), "unpacked_mb": round(raw / 1e6, 1),
            "packed_mb": round(pack_path.stat().st_size / 1e6, 1)}


def unpack(pack_path: Optional[Path] = None, index_path: Optional[Path] = None, force: bool = False) -> bool:
    """Write the local site pack out as the index. Leaves an existing index alone unless ``force``."""
    import gzip
    import shutil

    pack_path, index_path = pack_path or PACK_PATH, index_path or INDEX_PATH
    if not pack_path.is_file() or (index_path.is_file() and not force):
        return False
    _close_all()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_suffix(".unpacking")
    with gzip.open(pack_path, "rb") as gz, open(tmp, "wb") as fh:
        shutil.copyfileobj(gz, fh, 1 << 20)
    tmp.replace(index_path)
    return True


_state: dict = {}


def _close_all() -> None:
    for opened in _state.values():
        if opened:
            opened[0].close()
    _state.clear()


def _open(index_path: Optional[Path] = None):
    """``(connection, meta)`` for the index, or None when there is none (or it is an older shape).

    With no index at the default path, the local site pack (if present) is unpacked there first.
    """
    path = index_path or INDEX_PATH
    key = str(path)
    if key not in _state:
        _state[key] = None
        if index_path is None and not path.is_file():
            try:
                unpack()
            except OSError:
                pass
        if path.is_file():
            try:
                db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
                meta = dict(db.execute("SELECT key, value FROM meta"))
                if meta.get("version") == str(INDEX_VERSION):
                    _state[key] = (db, meta)
                else:
                    db.close()
            except sqlite3.Error:
                _state[key] = None
    return _state[key]


def _in_regions(meta: dict, lat: float, lon: float, margin_m: float) -> bool:
    dlat, dlon = _reach(lat, margin_m)
    return any(s <= lat - dlat and lat + dlat <= n and w <= lon - dlon and lon + dlon <= e
               for s, w, n, e in json.loads(meta.get("regions") or "{}").values())


def covers(lat: float, lon: float, margin_m: float = 0.0, index_path: Optional[Path] = None) -> bool:
    """True when the index holds the whole ``margin_m`` neighbourhood of the point.

    Full index: the neighbourhood lies inside an indexed region. Site pack: the point is within
    ``SITE_MATCH_M`` of a packed property and the neighbourhood is within the pack's reach of it.
    """
    opened = _open(index_path)
    if not opened:
        return False
    db, meta = opened
    if meta.get("coverage") != "sites":
        return _in_regions(meta, lat, lon, margin_m)
    if margin_m + SITE_MATCH_M > float(meta.get("reach_m") or 0.0):
        return False
    dlat, dlon = _reach(lat, SITE_MATCH_M)
    return db.execute("SELECT 1 FROM sites WHERE north >= ? AND south <= ? AND east >= ? AND west <= ? LIMIT 1",
                      (lat - dlat, lat + dlat, lon - dlon, lon + dlon)).fetchone() is not None


def fetch_site(lat: float, lon: float, radius_m: float = 150.0, index_path: Optional[Path] = None) -> Optional[dict]:
    """Everything indexed within ``radius_m`` of the point (by bounding box), or None when the point is not covered."""
    if not covers(lat, lon, radius_m, index_path):
        return None
    db, _meta = _open(index_path)
    dlat, dlon = _reach(lat, radius_m)
    rows = db.execute(
        "SELECT e.body FROM boxes b JOIN elements e ON e.rid = b.rid "
        "WHERE b.north >= ? AND b.south <= ? AND b.east >= ? AND b.west <= ? ORDER BY e.rid",
        (lat - dlat, lat + dlat, lon - dlon, lon + dlon)).fetchall()
    return {"elements": [json.loads(r[0]) for r in rows], "source": "local OpenStreetMap extract"}


def main(argv: list[str]) -> int:
    if "--download" in argv:
        download(force="--force" in argv)
        return 0
    if "--pack" in argv:
        print(export_pack(property_points()))
        return 0
    if "--unpack" in argv:
        done = unpack(force="--force" in argv)
        if done:
            print(f"unpacked {PACK_PATH.name} to {INDEX_PATH}")
        elif INDEX_PATH.is_file():
            print("nothing done: an index is already there (add --force to replace it)")
        else:
            print(f"nothing done: {PACK_PATH} was not found")
        return 0
    numbers = [a for a in argv if not a.startswith("--")]
    if len(numbers) == 2:
        lat, lon = float(numbers[0]), float(numbers[1])
        started = time.perf_counter()
        site = fetch_site(lat, lon)
        if site is None:
            print("not covered by the local index")
            return 1
        kinds: dict[str, int] = {}
        for e in site["elements"]:
            t = e.get("tags") or {}
            kind = "road" if "highway" in t else "place" if e["type"] == "node" else \
                "building" if "building" in t else "car park" if t.get("amenity") == "parking" else "residential land"
            kinds[kind] = kinds.get(kind, 0) + 1
        print(f"{len(site['elements'])} elements in {(time.perf_counter() - started) * 1000:.0f} ms: {kinds}")
        return 0
    print(build_index())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
