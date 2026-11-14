"""CoStar hackathon kit -> per-property facts used to assemble a ``Building``.

Reads the JSON Lines kit under ``<COSTAR_DATA_DIR>/costar-hackathon-kit`` (no
network). ``load_property(property_id)`` scans the large files once and caches
the small extract as ``data/processed/costar_<property_id>.json``.

What comes out (all areas in square feet, as CoStar reports them)::

    {"property_id", "name", "address", "latlng", "stories", "rba_sf",
     "typical_floor_sf", "slab_to_slab_ft", "core_factor_pct", "elevators",
     "parking_spaces", "floors": [{"floor": 4, "sf": 19272.0}, ...],
     "suites": [{"floor": 4, "suite": "420", "sf": 3499.0, "status": "occupied",
                 "tenant": "…" | None, "industry": "…" | None,
                 "employees": int | None, "available": bool}, ...],
     "matterport": [{"model_id": "MocRC6tyWqy", "caption": "Suite 420 Tour"}, ...]}

``status`` is ``"occupied"`` when a current tenancy (no ``toDate``) names the
suite, otherwise ``"available"`` (marketed in a listing with no current tenant).
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Iterator

from core.schema import Building

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
KIT_NAME = "costar-hackathon-kit"

MATTERPORT_REFERENCE_TYPES = (36, 234)  # MatterportScan, MatterportStreamingKey
OCCUPIED_FACTOR = 1.0
VACANT_FACTOR = 0.15  # marketed space with no current tenant: cleaners, brokers, contractors
UNKNOWN_FACTOR = 0.5  # floor area CoStar says nothing about


def costar_data_dir() -> Path:
    raw = os.environ.get("COSTAR_DATA_DIR", "./data/raw/costar")
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def kit_dir() -> Path:
    return costar_data_dir() / KIT_NAME


def _jsonl(name: str, needle: str | None = None) -> Iterator[dict]:
    """Records of one kit dataset; ``needle`` is a cheap substring pre-filter per line."""
    path = kit_dir() / "data" / f"{name}.jsonl"
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if needle is None or needle in line:
                yield json.loads(line)


def floor_number(floor_name_code: int | None) -> int | None:
    """CoStar ``FloorName`` enum -> storey number (FirstFloor = 24 -> 1); None if not a numbered floor."""
    if floor_name_code is None:
        return None
    c = int(floor_name_code)
    if c in (22, 23, 24):  # Lobby, Ground, FirstFloor
        return 1
    if c == 25:
        return 2
    if c == 27:
        return 3
    if c >= 28:  # FourthFloor = 28, FifthFloor = 29, ...
        return c - 24
    return None


def _suite_floor(suite: str | None) -> int | None:
    """Storey implied by a suite number ("420" -> 4, "1250" -> 12)."""
    m = re.match(r"\d+", suite or "")
    if not m or len(m.group()) < 3:
        return None
    return int(m.group()[:-2])


def _matterport_attachments(obj: Any) -> Iterator[dict]:
    if isinstance(obj, dict):
        if obj.get("videoKey") and obj.get("videoExternalReferenceType") in MATTERPORT_REFERENCE_TYPES:
            yield obj
        for value in obj.values():
            yield from _matterport_attachments(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _matterport_attachments(value)


def load_property(property_id: int, refresh: bool = False) -> dict:
    """Facts about one commercial property (see module docstring), cached on disk."""
    cache = PROCESSED_DIR / f"costar_{property_id}.json"
    if cache.is_file() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    if not kit_dir().is_dir():
        raise FileNotFoundError(f"CoStar kit not found at {kit_dir()}")

    pid_needles = (f'"id": {property_id},', f'"id":{property_id},')
    prop = next(
        (r for needle in pid_needles for r in _jsonl("commercial-property", needle) if r.get("id") == property_id),
        None,
    )
    if prop is None:
        raise KeyError(f"CoStar property {property_id} not found")

    addr = prop.get("primaryAddress") or {}
    office = prop.get("propertyOffice") or {}
    structural = office.get("structural") or {}
    floor_by_id: dict[int, int | None] = {}
    floors: list[dict] = []
    for f in prop.get("floors") or []:
        n = floor_number(f.get("floorName"))
        floor_by_id[f["propertyFloorId"]] = n
        if n is not None:
            floors.append({"floor": n, "sf": (f.get("floorSize") or {}).get("squareFeet")})
    floors.sort(key=lambda f: f["floor"])

    # Current tenancies, merged per suite (several records can name the same suite).
    suites: dict[tuple[int, str], dict] = {}
    tenant_ids: set[int] = set()
    for needle in (f'"propertyId": {property_id}', f'"propertyId":{property_id}'):
        for occ in _jsonl("location-occupancy", needle):
            if occ.get("propertyId") != property_id or occ.get("toDate"):
                continue
            for space in occ.get("spaces") or []:
                suite = (space.get("suite") or "").strip()
                floor = floor_by_id.get(space.get("propertyFloorId")) or _suite_floor(suite)
                if not suite or floor is None:
                    continue
                entry = suites.setdefault(
                    (floor, suite),
                    {"floor": floor, "suite": suite, "sf": 0.0, "status": "occupied", "tenant": None,
                     "industry": None, "employees": None, "available": False, "_tenant_id": None},
                )
                sf = float(space.get("squareFeetOccupied") or 0.0)
                if sf >= entry["sf"]:
                    entry["sf"] = sf
                    if occ.get("tenantLocationId"):
                        entry["_tenant_id"] = occ["tenantLocationId"]
                if entry["_tenant_id"] is None and occ.get("tenantLocationId"):
                    entry["_tenant_id"] = occ["tenantLocationId"]
                if entry["_tenant_id"]:
                    tenant_ids.add(entry["_tenant_id"])

    # Space marketed for lease.
    models: dict[str, str] = {}
    for needle in (f'"propertyId": {property_id}', f'"propertyId":{property_id}'):
        for listing in _jsonl("commercial-listing-summary", needle):
            if listing.get("propertyId") != property_id:
                continue
            for att in _matterport_attachments(listing):
                models.setdefault(att["videoKey"], att.get("caption") or "")
            for space in ((listing.get("forLeaseSummary") or {}).get("spaces")) or []:
                suite = (space.get("suite") or "").strip()
                floor = floor_number(space.get("floorName")) or _suite_floor(suite)
                if not suite or floor is None:
                    continue
                sf = float(space.get("squareFeetMax") or 0.0)
                entry = suites.setdefault(
                    (floor, suite),
                    {"floor": floor, "suite": suite, "sf": sf, "status": "available", "tenant": None,
                     "industry": None, "employees": None, "available": True, "_tenant_id": None},
                )
                entry["available"] = True
                entry["sf"] = max(entry["sf"], sf)

    for att in _matterport_attachments(prop):
        models.setdefault(att["videoKey"], att.get("caption") or "")
    starter = kit_dir() / "commercial-properties.csv"
    if starter.is_file():
        with open(starter, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if row.get("MatterportModelId") and int(row["PropertyId"]) == property_id:
                    models.setdefault(row["MatterportModelId"], "")

    # Who the tenants are.
    if tenant_ids:
        companies: dict[int, dict] = {}
        for rec in _jsonl("company-detail"):
            if rec.get("id") in tenant_ids:
                companies[rec["id"]] = rec
                if len(companies) == len(tenant_ids):
                    break
        for entry in suites.values():
            rec = companies.get(entry["_tenant_id"])
            if rec:
                profile = rec.get("profile") or {}
                entry["tenant"] = profile.get("companyName") or rec.get("name")
                entry["industry"] = profile.get("industryType")
                entry["employees"] = rec.get("numEmployees")
    for entry in suites.values():
        entry.pop("_tenant_id", None)

    street = " ".join(
        str(addr[k]) for k in ("streetNumber", "streetDirection", "streetName", "streetType") if addr.get(k)
    )
    out = {
        "property_id": property_id,
        "name": prop.get("buildingName") or street,
        "address": ", ".join(p for p in (street, addr.get("city"), addr.get("state")) if p),
        "latlng": [addr.get("latitude"), addr.get("longitude")],
        "property_type": prop.get("propertyType"),
        "stories": prop.get("numStoriesAboveGrade"),
        "rba_sf": prop.get("rentableBuildingAreaSquareFeet"),
        "typical_floor_sf": prop.get("grossTypicalFloorSquareFeet"),
        "slab_to_slab_ft": structural.get("slabToSlabHeightFeet"),
        "core_factor_pct": office.get("coreFactor"),
        "elevators": sum(int(e.get("numElevators") or 0) for e in structural.get("elevators") or []),
        "parking_spaces": sum(int(p.get("totalSpaces") or 0) for p in prop.get("parking") or []),
        "floors": floors,
        "suites": sorted(suites.values(), key=lambda s: (s["floor"], s["suite"])),
        "matterport": [{"model_id": k, "caption": v} for k, v in sorted(models.items())],
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def suite_occupancy_factor(suite: dict | None) -> float:
    """``Room.occupancy_factor`` for a suite record from ``load_property``."""
    if suite is None:
        return UNKNOWN_FACTOR
    return OCCUPIED_FACTOR if suite.get("status") == "occupied" else VACANT_FACTOR


def apply_occupancy(building: Building, factors: dict[str, float]) -> Building:
    """Return a copy of ``building`` with ``Room.occupancy_factor`` overridden."""
    out = building.model_copy(deep=True)
    for room in out.rooms:
        if room.id in factors:
            room.occupancy_factor = min(1.0, max(0.0, float(factors[room.id])))
    return out


if __name__ == "__main__":
    import sys

    p = load_property(int(sys.argv[1]), refresh="--refresh" in sys.argv)
    print({k: v for k, v in p.items() if k not in ("suites", "matterport")})
    for s in p["suites"]:
        print(f"  F{s['floor']} suite {s['suite']:<5} {s['sf']:>7.0f} sf  {s['status']:<9} "
              f"{'listed ' if s['available'] else '       '}{s['tenant'] or ''} {('(' + str(s['industry']) + ')') if s['industry'] else ''}")
    print("matterport:", p["matterport"])
