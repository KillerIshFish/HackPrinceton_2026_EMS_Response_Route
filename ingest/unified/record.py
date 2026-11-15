"""Property record -> the facts the generator needs (metric, missing fields tolerated).

Everything is read with ``.get()``: the kit omits fields it has no value for.
Areas arrive in square feet and leave in square metres; nothing downstream sees
imperial units.
"""

from __future__ import annotations

from typing import Optional

from ingest.costar import MATTERPORT_REFERENCE_TYPES, floor_number
from ingest.unified import index

SQFT_M2 = 0.09290304  # exact: 0.3048^2 m2 per square foot
FT_M = 0.3048  # exact: metres per international foot

# CoStar propertyType / apartmentStyleType codes -> this pipeline's vocabulary; unlisted codes become "other" / "".
USE_TYPES = {1: "hotel", 2: "industrial", 5: "office", 6: "retail", 7: "flex", 9: "mixed_use",
             11: "apartment", 13: "healthcare"}
APARTMENT_STYLES = {1: "garden", 2: "mid_rise", 3: "high_rise", 4: "low_rise", 5: "house", 6: "townhome"}


def _m2(sqft) -> Optional[float]:
    try:
        value = float(sqft)
    except (TypeError, ValueError):
        return None
    return value * SQFT_M2 if value > 0 else None


def _scan_ids(attachments) -> list[str]:
    out: list[str] = []
    for a in attachments or []:
        key = a.get("videoKey")
        if key and a.get("videoExternalReferenceType") in MATTERPORT_REFERENCE_TYPES and key not in out:
            out.append(str(key))
    return out


def _street(addr: dict) -> str:
    return " ".join(str(addr[k]) for k in ("streetNumber", "streetDirection", "streetName", "streetType") if addr.get(k))


def property_facts(property_id: int) -> dict:
    """Normalised facts for one property (see the keys below). Raises KeyError for an unknown id."""
    prop = index.get_record(property_id)
    addr = prop.get("primaryAddress") or {}
    apt = prop.get("propertyApartment") or {}
    office = prop.get("propertyOffice") or {}
    structural = office.get("structural") or {}
    street = _street(addr)

    floor_by_id: dict[int, Optional[int]] = {}
    floor_area: dict[int, float] = {}
    for f in prop.get("floors") or []:
        n = floor_number(f.get("floorName"))
        if f.get("propertyFloorId") is not None:
            floor_by_id[f["propertyFloorId"]] = n
        area = _m2((f.get("floorSize") or {}).get("squareFeet"))
        if n is not None and area:
            floor_area[n] = area

    units: list[dict] = []
    for m in apt.get("models") or []:
        model_scans = _scan_ids(m.get("attachments"))
        for u in m.get("units") or []:
            number = str(u.get("unitNumber") or u.get("unitName") or "").strip()
            if not number:
                continue
            units.append({
                "id": str(u.get("id") or ""),
                "number": number,
                "building": str(u.get("buildingName") or "").strip(),
                "type": str(m.get("modelName") or m.get("id") or "").strip(),
                "beds": u.get("numBeds") if u.get("numBeds") is not None else m.get("numBeds"),
                "baths": u.get("numBaths") if u.get("numBaths") is not None else m.get("numBaths"),
                "area_m2": _m2(u.get("minSquareFeet")) or _m2(m.get("minSquareFeet")),
                "scans": _scan_ids(u.get("attachments")) or model_scans,
            })

    tenants: list[dict] = []
    if not apt:
        tenancies = index.get_tenancies(property_id)
        companies = index.get_companies(t.get("tenantLocationId") for t in tenancies)
        merged: dict[tuple, dict] = {}
        for occ in tenancies:
            company = companies.get(occ.get("tenantLocationId")) or {}
            name = ((company.get("profile") or {}).get("companyName") or company.get("name") or "").strip()
            for space in occ.get("spaces") or []:
                suite = str(space.get("suite") or "").strip()
                floor = floor_by_id.get(space.get("propertyFloorId"))
                key = (floor, suite or f"t{occ.get('tenantLocationId') or occ.get('id')}")
                entry = merged.setdefault(key, {"floor": floor, "suite": suite, "tenant": name, "area_m2": 0.0,
                                                "id": str(space.get("id") or occ.get("id") or "")})
                entry["area_m2"] = max(entry["area_m2"], _m2(space.get("squareFeetOccupied")) or 0.0)
                if name and not entry["tenant"]:
                    entry["tenant"] = name
        tenants = [merged[k] for k in sorted(merged, key=lambda k: (k[0] is None, k[0] or 0, str(k[1])))]

    elevators = list(apt.get("elevators") or []) + list(structural.get("elevators") or [])
    has_elevator: Optional[bool] = None
    if elevators:
        has_elevator = any(int(e.get("numElevators") or 0) > 0 or e.get("elevatorState") not in (None, 0)
                           for e in elevators)
    if apt.get("isWalkup") is True:
        has_elevator = False

    stories = int(prop.get("numStoriesAboveGrade") or 0)
    gross = _m2(prop.get("grossBuildingAreaSquareFeet")) or _m2(prop.get("rentableBuildingAreaSquareFeet"))
    typical = _m2(prop.get("grossTypicalFloorSquareFeet"))
    slab = structural.get("slabToSlabHeightFeet")
    return {
        "property_id": int(prop["id"]),
        "name": (prop.get("buildingName") or street or f"Property {prop['id']}").strip(),
        "address": ", ".join(p for p in (street, addr.get("city"), addr.get("state")) if p),
        "latlng": [addr.get("latitude"), addr.get("longitude")]
        if addr.get("latitude") is not None and addr.get("longitude") is not None else None,
        "use_type": USE_TYPES.get(prop.get("propertyType"), "other"),
        "style": APARTMENT_STYLES.get(apt.get("apartmentStyleType"), ""),
        "stories": stories,
        "gross_m2": gross,
        "typical_floor_m2": typical or (gross / stories if gross and stories else None),
        "floor_area_m2": floor_area,
        "storey_height_m": float(slab) * FT_M if slab else None,
        "num_buildings": int(apt.get("numBuildings") or 1),
        "total_units": int(apt.get("totalNumUnits") or 0),
        "has_elevator": has_elevator,
        "units": units,
        "tenants": tenants,
        "property_scans": _scan_ids(prop.get("attachments")),
    }
