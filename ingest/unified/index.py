"""Address / property-id lookup over the CoStar kit without loading it.

The property file is ~190 MB of JSON Lines. ``build_index()`` reads it once and
writes ``data/processed/property_index.sqlite`` holding, per property, the byte
offset of its line plus a few searchable columns; and the same for its current
tenancy records and tenant companies. At call time a lookup is one indexed SQLite
query and ``get_record()`` is one seek and one ``json.loads`` of a single line.

The index is a locator, not a model: no building is generated here. It is rebuilt
automatically when a source file's size or modification time changes.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Iterator, Optional

from ingest.costar import kit_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = REPO_ROOT / "data" / "processed" / "property_index.sqlite"
INDEX_VERSION = "3"  # part of the signature: bump when the table layout changes so old indexes are rebuilt
SOURCES = ("commercial-property", "location-occupancy", "company-detail")

_ABBREVIATIONS = {
    "street": "st", "avenue": "ave", "av": "ave", "boulevard": "blvd", "drive": "dr", "road": "rd",
    "lane": "ln", "court": "ct", "place": "pl", "parkway": "pkwy", "highway": "hwy", "circle": "cir",
    "terrace": "ter", "trail": "trl", "freeway": "fwy", "square": "sq", "plaza": "plz", "way": "way",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
}
_DIRECTIONS = {"n", "s", "e", "w", "ne", "nw", "se", "sw"}


def normalize(text: str) -> str:
    """"11900 West Olympic Boulevard, Los Angeles" -> "11900 w olympic blvd los angeles"."""
    tokens = re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).split()
    return " ".join(_ABBREVIATIONS.get(t, t) for t in tokens)


def _source_path(name: str) -> Path:
    return kit_dir() / "data" / f"{name}.jsonl"


def _signature() -> str:
    parts = [INDEX_VERSION]
    for name in SOURCES:
        p = _source_path(name)
        st = p.stat() if p.is_file() else None
        parts.append(f"{name}:{st.st_size if st else 0}:{int(st.st_mtime) if st else 0}")
    return "|".join(parts)


def _lines(path: Path) -> Iterator[tuple[int, bytes]]:
    """(byte offset, raw line) for every line of a file."""
    offset = 0
    with open(path, "rb") as fh:
        for line in fh:
            yield offset, line
            offset += len(line)


def _street(addr: dict, with_direction: bool = True) -> str:
    keys = ("streetNumber", "streetDirection", "streetName", "streetType") if with_direction else (
        "streetNumber", "streetName", "streetType")
    return " ".join(str(addr[k]) for k in keys if addr.get(k))


def _address_keys(prop: dict) -> set[str]:
    keys: set[str] = set()
    for addr in [prop.get("primaryAddress") or {}] + list(prop.get("addresses") or []):
        for with_dir in (True, False):
            street = normalize(_street(addr, with_dir))
            if street:
                keys.add(street)
    name = normalize(prop.get("buildingName") or "")
    if name:
        keys.add("name:" + name)
    for alias in prop.get("aliases") or []:
        text = normalize(alias.get("name") if isinstance(alias, dict) else alias)
        if text:
            keys.add("name:" + text)
    return keys


def build_index(path: Path = INDEX_PATH) -> dict:
    """(Re)build the index. Returns ``{"properties", "tenancies", "companies", "seconds"}``."""
    if not _source_path("commercial-property").is_file():
        raise FileNotFoundError(f"CoStar kit not found at {kit_dir()}")
    started = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".building")
    if tmp.exists():
        tmp.unlink()
    db = sqlite3.connect(tmp)
    db.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE properties (id INTEGER PRIMARY KEY, offset INTEGER, length INTEGER, name TEXT,
            address TEXT, city TEXT, state TEXT, postcode TEXT, lat REAL, lng REAL,
            property_type INTEGER, stories INTEGER, units INTEGER);
        CREATE TABLE address_keys (key TEXT, property_id INTEGER);
        CREATE TABLE occupancy (property_id INTEGER, offset INTEGER, length INTEGER);
        CREATE TABLE companies (id INTEGER PRIMARY KEY, offset INTEGER, length INTEGER);
        """
    )
    n_props = 0
    for offset, line in _lines(_source_path("commercial-property")):
        prop = json.loads(line)
        addr = prop.get("primaryAddress") or {}
        street = _street(addr)
        apt = prop.get("propertyApartment") or {}
        db.execute(
            "INSERT OR REPLACE INTO properties VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (prop["id"], offset, len(line), prop.get("buildingName") or street, street, addr.get("city"),
             addr.get("state"), addr.get("postcode"), addr.get("latitude"), addr.get("longitude"),
             prop.get("propertyType"), prop.get("numStoriesAboveGrade"), apt.get("totalNumUnits")),
        )
        db.executemany("INSERT INTO address_keys VALUES (?,?)", [(k, prop["id"]) for k in _address_keys(prop)])
        n_props += 1
    n_occ = 0
    occ_path = _source_path("location-occupancy")
    if occ_path.is_file():
        rows = []
        for offset, line in _lines(occ_path):
            # cheap pre-filter: past tenancies carry a toDate and are never read
            if b'"toDate"' in line and json.loads(line).get("toDate"):
                continue
            m = re.search(rb'"propertyId":\s*(\d+)', line)
            if m:
                rows.append((int(m.group(1)), offset, len(line)))
        db.executemany("INSERT INTO occupancy VALUES (?,?,?)", rows)
        n_occ = len(rows)
    n_co = 0
    co_path = _source_path("company-detail")
    if co_path.is_file():
        rows = []
        for offset, line in _lines(co_path):
            rec_id = json.loads(line).get("id")
            if rec_id is not None:
                rows.append((int(rec_id), offset, len(line)))
        db.executemany("INSERT OR REPLACE INTO companies VALUES (?,?,?)", rows)
        n_co = len(rows)
    db.executescript(
        """
        CREATE INDEX address_keys_key ON address_keys (key);
        CREATE INDEX occupancy_property ON occupancy (property_id);
        """
    )
    db.execute("INSERT INTO meta VALUES ('signature', ?)", (_signature(),))
    db.commit()
    db.close()
    if path.exists():
        path.unlink()
    tmp.rename(path)
    return {"properties": n_props, "tenancies": n_occ, "companies": n_co,
            "seconds": round(time.perf_counter() - started, 2)}


_connection: Optional[sqlite3.Connection] = None


def connect(path: Path = INDEX_PATH) -> sqlite3.Connection:
    """Open the index, building it first when it is missing or stale."""
    global _connection
    if _connection is not None:
        return _connection
    fresh = False
    if path.is_file():
        try:
            db = sqlite3.connect(path, check_same_thread=False)
            row = db.execute("SELECT value FROM meta WHERE key = 'signature'").fetchone()
            fresh = bool(row) and row[0] == _signature()
            if not fresh:
                db.close()
        except sqlite3.Error:
            fresh = False
    if not fresh:
        build_index(path)
        db = sqlite3.connect(path, check_same_thread=False)
    db.row_factory = sqlite3.Row
    _connection = db
    return db


def _row(row: sqlite3.Row) -> dict:
    out = {k: row[k] for k in ("id", "name", "address", "city", "state", "postcode", "lat", "lng",
                               "property_type", "stories", "units")}
    out["property_id"] = out.pop("id")
    return out


def lookup(query: str, limit: int = 8) -> list[dict]:
    """Properties matching an address, a building name or a property id, best first."""
    db = connect()
    q = str(query).strip()
    if re.fullmatch(r"\d{4,}", q):  # a bare property id (4+ digits; shorter numbers read as street numbers)
        row = db.execute("SELECT * FROM properties WHERE id = ?", (int(q),)).fetchone()
        if row:
            return [_row(row)]
    norm = normalize(q.split(",")[0]) or normalize(q)
    full = normalize(q)
    if not full:
        return []
    ids: list[int] = []

    def add(rows) -> None:
        for rw in rows:
            if rw[0] not in ids:
                ids.append(rw[0])

    add(db.execute("SELECT property_id FROM address_keys WHERE key IN (?, ?)", (norm, "name:" + norm)))
    # "11900 w olympic blvd los angeles ca 90064": drop trailing tokens until a street matches
    tokens = full.split()
    for end in range(len(tokens) - 1, 1, -1):
        if ids:
            break
        prefix = " ".join(tokens[:end])
        add(db.execute("SELECT property_id FROM address_keys WHERE key IN (?, ?)", (prefix, "name:" + prefix)))
    if len(ids) < limit and tokens:
        # same street number, most shared words first; or a building-name fragment
        head = tokens[0]
        like = (head + " %") if head.isdigit() else ("name:%" + norm + "%")
        scored = []
        # LIMIT 400 bounds the fuzzy scan; ranking is by shared words, then the shorter key
        for rw in db.execute("SELECT property_id, key FROM address_keys WHERE key LIKE ? LIMIT 400", (like,)):
            overlap = len(set(tokens) & set(rw[1].replace("name:", "").split()))
            scored.append((-overlap, len(rw[1]), rw[0]))
        for _, _, pid in sorted(scored):
            if pid not in ids:
                ids.append(pid)
    out = []
    for pid in ids[:limit]:
        row = db.execute("SELECT * FROM properties WHERE id = ?", (pid,)).fetchone()
        if row:
            out.append(_row(row))
    return out


def _read(name: str, offset: int, length: int) -> dict:
    with open(_source_path(name), "rb") as fh:
        fh.seek(offset)
        return json.loads(fh.read(length))


def get_record(property_id: int) -> dict:
    """The full property record: one seek, one line."""
    row = connect().execute("SELECT offset, length FROM properties WHERE id = ?", (int(property_id),)).fetchone()
    if row is None:
        raise KeyError(f"property {property_id} is not in the index")
    return _read("commercial-property", row[0], row[1])


def get_tenancies(property_id: int) -> list[dict]:
    """Current tenancy records of a property."""
    rows = connect().execute("SELECT offset, length FROM occupancy WHERE property_id = ? ORDER BY offset",
                             (int(property_id),)).fetchall()
    return [_read("location-occupancy", o, n) for o, n in rows]


def get_companies(ids) -> dict[int, dict]:
    db = connect()
    out: dict[int, dict] = {}
    for cid in sorted({int(i) for i in ids if i is not None}):
        row = db.execute("SELECT offset, length FROM companies WHERE id = ?", (cid,)).fetchone()
        if row:
            out[cid] = _read("company-detail", row[0], row[1])
    return out


def all_property_ids() -> list[int]:
    return [rw[0] for rw in connect().execute("SELECT id FROM properties ORDER BY id")]
