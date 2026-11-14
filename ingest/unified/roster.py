"""Layers 1-3: floor stack, and every unit or tenant placed on a floor.

Unit numbers usually encode the floor, but in several ways. ``infer_scheme``
tries each known scheme against all of a property's unit numbers and keeps the
first that explains at least 85% of them:

    floor_hundreds     "304"      -> floor 3            (also "B-304", "01-0317" with a building prefix)
    floor_first_digit  "7156"     -> floor 7            (four digits, one-digit floor)
    building_floor     "2314"     -> building 2, floor 3
    building_hundreds  "1807"     -> building 18, floor not encoded
    sequential         "03", "A"  -> floor not encoded

When the floor is not encoded, units are spread evenly over the floors in number
order and marked ``floor_source = "spread"``; a caller's own floor statement or the
phone's height fix then does the work the number cannot.

An LLM could decode schemes these rules miss. That is deliberately not called
here: ``scheme_override`` accepts a cached answer (a scheme name per property), so
the same inputs always give the same roster.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Optional

DEFAULT_STOREY_M = {"apartment": 3.0, "hotel": 3.2, "office": 4.0, "retail": 4.5, "industrial": 7.0,
                    "flex": 5.0, "healthcare": 4.2, "mixed_use": 3.6, "other": 3.5}
SCHEME_MIN_SHARE = 0.85
STOREY_TOLERANCE = 2  # unit numbers may name a floor this far above the recorded storey count
SCHEMES = ("floor_hundreds", "floor_first_digit", "building_floor", "building_hundreds", "sequential")

_PREFIXED = re.compile(r"([A-Za-z0-9]+)\s*[-–]\s*0*(\d+)\s*[A-Za-z]?$")
_PLAIN = re.compile(r"([A-Za-z]?)(\d+)([A-Za-z]?)$")


def storey_height(facts: dict) -> tuple[float, str]:
    if facts.get("storey_height_m"):
        return round(float(facts["storey_height_m"]), 3), "record"
    return DEFAULT_STOREY_M.get(facts.get("use_type") or "other", 3.5), "default"


def parse_number(raw: str) -> tuple[str, Optional[int], str]:
    """``(building prefix, number, tidy label)`` from "709 S. Mariposa Avenue - #105", "01-0317", "B304"."""
    text = str(raw).strip()
    if "#" in text:
        text = text.split("#")[-1].strip()
    m = _PREFIXED.search(text)
    if m and len(m.group(1)) <= 3:
        return m.group(1).lstrip("0") or "0", int(m.group(2)), text
    tail = text.split(" - ")[-1].split()[-1] if text.split() else text
    m = _PLAIN.search(tail)
    if m:
        return "", int(m.group(2)), tail
    return "", None, text


def _decode(number: int, scheme: str) -> tuple[str, Optional[int]]:
    """``(building, floor)`` a scheme reads out of a number; floor None when not encoded."""
    if scheme == "floor_hundreds":
        return "", number // 100 if number >= 100 else None
    if scheme == "floor_first_digit":
        return "", int(str(number)[0]) if number >= 1000 else None
    if scheme == "building_floor":
        return (str(number // 1000), (number // 100) % 10) if number >= 1000 else ("", None)
    if scheme == "building_hundreds":
        return (str(number // 100), None) if number >= 100 else ("", None)
    return "", None


def infer_scheme(numbers: list[int], stories: int, num_buildings: int) -> str:
    """The first scheme that explains enough of ``numbers`` for a building of ``stories`` floors."""
    if not numbers:
        return "sequential"
    top = max(stories, 1)

    def share(scheme: str) -> float:
        ok = 0
        for n in numbers:
            _, floor = _decode(n, scheme)
            ok += floor is not None and 1 <= floor <= top
        return ok / len(numbers)

    if share("floor_hundreds") >= SCHEME_MIN_SHARE:
        return "floor_hundreds"
    if top <= 9 and share("floor_first_digit") >= SCHEME_MIN_SHARE:
        return "floor_first_digit"
    # Door numbers that run a floor or two past the recorded storey count ("2xx" to "6xx" in a
    # building listed with five storeys: a lobby or parking level the record does not count).
    # The numbers are what is painted on the doors, so they are trusted over the count.
    top += STOREY_TOLERANCE
    if share("floor_hundreds") >= SCHEME_MIN_SHARE:
        return "floor_hundreds"
    if top <= 9 and share("floor_first_digit") >= SCHEME_MIN_SHARE:
        return "floor_first_digit"
    top -= STOREY_TOLERANCE
    if num_buildings > 1 and share("building_floor") >= SCHEME_MIN_SHARE:
        return "building_floor"
    if num_buildings > 1 and sum(n >= 100 for n in numbers) / len(numbers) >= SCHEME_MIN_SHARE:
        return "building_hundreds"
    return "sequential"


def _spread(items: list[dict], floors: list[int]) -> None:
    """Assign floors evenly, in label order, to items whose floor is unknown."""
    todo = sorted((u for u in items if u["floor"] is None), key=lambda u: (u["sort"], u["label"], u["id"]))
    if not todo or not floors:
        return
    per = math.ceil(len(todo) / len(floors))
    for i, u in enumerate(todo):
        u["floor"] = floors[min(i // per, len(floors) - 1)]
        u["floor_source"] = "spread"


def build_roster(facts: dict, scheme_override: Optional[str] = None) -> dict:
    """``{"stories", "storey_height_m", "height_source", "scheme", "slots": [...], "stats": {...}}``.

    Each slot: ``{"id", "kind": "unit" | "tenant", "label", "block", "floor", "floor_source",
    "area_m2", "beds", "baths", "type", "scans", "tenant"}``.
    """
    stories = int(facts.get("stories") or 0)
    slots: list[dict] = []
    scheme = "none"

    units = facts.get("units") or []
    if units:
        parsed = [parse_number(u["number"]) for u in units]
        numbers = [n for _, n, _ in parsed if n is not None]
        prefixed = sum(1 for p, _, _ in parsed if p) / len(parsed)
        scheme = scheme_override if scheme_override in SCHEMES else infer_scheme(
            numbers, stories or 1, int(facts.get("num_buildings") or 1))
        named = sum(1 for u in units if u.get("building") and u["building"].upper() != "N/A") / len(units)
        seen: Counter = Counter()
        if scheme in ("floor_hundreds", "floor_first_digit") and stories:
            # the highest floor the numbers name, when it is within tolerance of the record
            named_floors = [f for f in (_decode(n, scheme)[1] for n in numbers) if f and f <= stories + STOREY_TOLERANCE]
            stories = max(stories, max(named_floors, default=stories))
        for u, (prefix, number, label) in zip(units, parsed):
            building, floor = _decode(number, scheme) if number is not None else ("", None)
            if named >= SCHEME_MIN_SHARE:
                block = u["building"]
            elif prefixed >= SCHEME_MIN_SHARE:
                block = prefix
            else:
                block = building
            source = "number" if floor is not None else ""
            if floor is not None and stories and not 1 <= floor <= stories:
                floor, source = min(max(floor, 1), stories), "clamped"
            seen[(block, label)] += 1
            uid = u.get("id") or f"{block}:{label}:{seen[(block, label)]}"
            slots.append({
                "id": f"u{uid}", "kind": "unit", "label": label, "block": str(block or ""), "floor": floor,
                "floor_source": source, "area_m2": u.get("area_m2"), "beds": u.get("beds"), "baths": u.get("baths"),
                "type": u.get("type") or "", "scans": list(u.get("scans") or []), "tenant": "",
                "sort": number if number is not None else 10 ** 9,
            })
    elif facts.get("total_units"):
        # a unit count but no unit list: numbered placeholders, spread over the floors
        scheme = "synthesised"
        for i in range(int(facts["total_units"])):
            slots.append({"id": f"s{i + 1}", "kind": "unit", "label": f"Unit {i + 1}", "block": "", "floor": None,
                          "floor_source": "", "area_m2": None, "beds": None, "baths": None, "type": "",
                          "scans": [], "tenant": "", "sort": i})

    for t in facts.get("tenants") or []:
        label = t.get("suite") or t.get("tenant") or "Tenant"
        if t.get("suite") and t.get("tenant"):
            label = f"{t['suite']} {t['tenant']}"
        number = parse_number(t.get("suite") or "")[1]
        slots.append({
            "id": f"t{t.get('id') or len(slots)}", "kind": "tenant", "label": label[:60], "block": "",
            "floor": t.get("floor"), "floor_source": "record" if t.get("floor") else "",
            "area_m2": t.get("area_m2") or None, "beds": None, "baths": None, "type": "", "scans": [],
            "tenant": t.get("tenant") or "", "sort": number if number is not None else 10 ** 9,
        })

    if not stories:
        stories = max([s["floor"] for s in slots if s["floor"]] or [1])
    floors = list(range(1, stories + 1))
    by_block: dict[str, list[dict]] = defaultdict(list)
    for s in slots:
        if s["floor"] is not None and not 1 <= s["floor"] <= stories:
            s["floor"], s["floor_source"] = min(max(s["floor"], 1), stories), "clamped"
        by_block[s["block"]].append(s)
    for items in by_block.values():
        _spread(items, floors)

    height, height_source = storey_height(facts)
    sources = Counter(s["floor_source"] for s in slots)
    return {
        "stories": stories, "storey_height_m": height, "height_source": height_source, "scheme": scheme,
        "slots": sorted(slots, key=lambda s: (s["block"], s["floor"], s["sort"], s["label"], s["id"])),
        "stats": {
            "slots": len(slots), "units": sum(s["kind"] == "unit" for s in slots),
            "tenants": sum(s["kind"] == "tenant" for s in slots), "blocks": len(by_block) or 1,
            "floor_from_number": sources.get("number", 0), "floor_from_record": sources.get("record", 0),
            "floor_spread": sources.get("spread", 0), "floor_clamped": sources.get("clamped", 0),
            "storeys_recorded": int(facts.get("stories") or 0),
            "scanned_unit_types": len({(s["type"], tuple(s["scans"])) for s in slots if s["scans"]}),
            "units_with_scan": sum(1 for s in slots if s["scans"]),
        },
    }
