"""Match values for Evidence items: how well each room fits one statement.

* ``kind="vocab"``    -> ``core.clue_likelihood.clue_match`` (exactly the older numbers). One
  addition: a ``floor`` clue with value ``basement`` selects the rooms on the building's
  below-ground floors (``basement_floors``); in a building without one it is unmatched.
* ``kind="grounded"`` -> rooms satisfying *every* set condition of the ``RoomSelector``
  get 1.0 (0.0 with ``evidence.negate``), every other room ``evidence.default_match``.

RoomSelector conditions (all optional, AND-combined):

``room_ids``         room id in the list
``floor_labels``     ``Room.floor_label`` in the list (case / space insensitive)
``room_types``       ``Room.room_type`` in the list (``"Open office"`` == ``"open_office"``)
``name_contains``    case-insensitive substring of ``Room.name``
``unit``             the room's label equals the unit (case-insensitive, "#",
                     "suite"/"unit"/"apt" prefixes and leading zeros ignored). A room's
                     labels are its sidecar ``rooms[room_id]["suite"]`` and the numbers
                     printed in its name ("Office 302" -> 302, "Conference Room 2N" -> 2n,
                     "340 Citrin Cooperman" -> 340; see ``name_labels``). Suite records
                     win: name labels are only used when no room has that suite.
``tenant_contains``  case-insensitive substring of sidecar ``rooms[room_id]["tenant"]``
``has_windows``, ``window_facing``, ``near_elevator``, ``near_stairs``
                     equality with the Room field
``within_m_of``      walking distance (sum of edge ``length_m`` on ``core.building_graph``,
                     same-floor edges only) from ``room_id`` <= ``radius_m``; the anchor
                     room itself is included (distance 0)

A selector that matches no room (or sets no condition) is uninformative: the match is
1.0 everywhere and ``EvidenceMatch.unmatched`` is True, so the UI can say the statement
could not be placed. The posterior is then unchanged by it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping

import networkx as nx

from core.building_graph import build_graph
from core.clue_likelihood import clue_match
from core.schema import Building, Evidence, Room, RoomSelector

# A vocab room counts towards ``matched_room_ids`` (for the UI outline) above this match.
VOCAB_MATCHED_ABOVE = 0.5

# Edge kinds that stay on one floor (``within_m_of`` walks only these).
_SAME_FLOOR_KINDS = ("door", "corridor")


@dataclass(frozen=True)
class EvidenceMatch:
    """Result of ``evidence_match``.

    * ``match``            room id -> match value in [0, 1] (every room of the building).
    * ``unmatched``        True when a grounded selector matched no room; ``match`` is
                           then 1.0 everywhere (uninformative) and the UI says the
                           statement could not be placed.
    * ``matched_room_ids`` rooms the statement points at (selector hits; for vocab
                           clues, rooms with match above ``VOCAB_MATCHED_ABOVE``).
    """

    match: dict[str, float]
    unmatched: bool = False
    matched_room_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sidecar metadata (suite / tenant per room)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _load_room_meta_cached(building_id: str) -> dict[str, dict]:
    from ingest.real_building import load_meta  # local import: ingest pulls in heavier deps

    rooms = load_meta(building_id).get("rooms") or {}
    return {str(k): dict(v or {}) for k, v in rooms.items()}


def room_meta(building: Building) -> dict[str, dict]:
    """``room id -> sidecar metadata dict`` (``suite``, ``tenant``, ...); {} for synthetic buildings."""
    return _load_room_meta_cached(building.id)


def has_unit_metadata(building: Building, meta: Mapping[str, Mapping[str, Any]] | None = None) -> bool:
    """True when at least one room of the building has a ``suite`` in its sidecar metadata."""
    meta = room_meta(building) if meta is None else meta
    return any((meta.get(r.id) or {}).get("suite") for r in building.rooms)


_UNIT_PREFIX = re.compile(r"^(?:#|no\.?|number|suite|ste\.?|unit|apartment|apt\.?|room|rm\.?|office)\s*#?\s*", re.IGNORECASE)


def normalize_unit(value: object) -> str:
    """Canonical unit string: ``"Suite #0304"`` -> ``"304"``, ``"12b"`` -> ``"12b"``."""
    s = str(value or "").strip().lower()
    s = _UNIT_PREFIX.sub("", s).replace(" ", "").replace("-", "")
    m = re.fullmatch(r"0*(\d+)([a-z]?)", s)
    if m:
        return m.group(1) + m.group(2)
    return s


def _norm_token(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


# A token with a digit in a room name is a label printed on that room: "Office 302",
# "Conference Room 2N", "Unit 202", "340 Citrin Cooperman". Numbers that name the floor,
# the building or the parking level ("Floor 2 (no unit data)", "Stair A (bldg 11)") are not.
_NAME_LABEL = re.compile(r"(?<![\w.#-])#?(\d{1,5}(?:-\d{1,5})?[A-Za-z]?|[A-Za-z]\d{1,4})(?![\w.]|,\d|-\d)")
_NOT_A_LABEL_AFTER = re.compile(
    r"(?<!\w)(?:floors?|levels?|fl\.?|lvl\.?|bldg\.?|buildings?|towers?|wings?|phases?|blocks?)\s*#?\s*$",
    re.IGNORECASE,
)


@lru_cache(maxsize=4096)
def name_labels(name: str) -> frozenset[str]:
    """Unit-like labels printed in a room's name, normalised with ``normalize_unit``.

    ``"Office 302"`` -> {"302"}, ``"Conference Room 2N"`` -> {"2n"}, ``"Suite 860 office 1"``
    -> {"860", "1"}, ``"Floor 2 (no unit data)"`` -> {} (a floor number is not a room label).
    """
    out: set[str] = set()
    for m in _NAME_LABEL.finditer(name or ""):
        if _NOT_A_LABEL_AFTER.search(name[: m.start()]):
            continue
        out.add(normalize_unit(m.group(1)))
    return frozenset(out)


def room_labels(building: Building, meta: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, frozenset[str]]:
    """``room id -> labels`` a caller could read off the door: the sidecar suite number and the
    numbers in the room's name. Rooms without any label are left out."""
    meta = room_meta(building) if meta is None else meta
    out: dict[str, frozenset[str]] = {}
    for r in building.rooms:
        labels = set(name_labels(r.name))
        suite = (meta.get(r.id) or {}).get("suite")
        if suite:
            labels.add(normalize_unit(suite))
        if labels:
            out[r.id] = frozenset(labels)
    return out


def unit_room_ids(building: Building, unit: object, meta: Mapping[str, Mapping[str, Any]] | None = None) -> list[str]:
    """Rooms labelled ``unit`` (building order): rooms whose sidecar suite equals it, or, when no
    room has that suite, rooms with that number printed in their name. Several rooms may share
    a label (the rooms of one suite, "Unit 302" in two buildings of a complex); all are returned."""
    u = normalize_unit(unit)
    if not u:
        return []
    meta = room_meta(building) if meta is None else meta
    by_suite = [r.id for r in building.rooms
                if (meta.get(r.id) or {}).get("suite") and normalize_unit(meta[r.id]["suite"]) == u]
    if by_suite:
        return by_suite
    return [r.id for r in building.rooms if u in name_labels(r.name)]


# ---------------------------------------------------------------------------
# "Basement": a floor a caller names without a number
# ---------------------------------------------------------------------------

BASEMENT_VALUES = ("basement", "cellar", "lower_level")
_BELOW_GROUND_LABEL = re.compile(
    r"^(?:b\d*|bsmt|basement|ll\d*|lower[ _-]?level|cellar|sub[ _-]?basement|sb\d*|-\d+)$", re.IGNORECASE)


def basement_floors(building: Building) -> list[int]:
    """Floor indices of the building's below-ground floors: a floor label such as "B", "B1",
    "LL", "Basement" or "-1", or a negative floor index. Empty for a building without one."""
    out: set[int] = set()
    for r in building.rooms:
        if r.floor < 0 or _BELOW_GROUND_LABEL.match(str(r.floor_label).strip()):
            out.add(r.floor)
    return sorted(out)


def is_basement_clue(clue: object) -> bool:
    """True for a ``floor`` vocabulary clue whose value is "basement" (or "cellar", "lower_level")."""
    return (getattr(clue, "type", None) == "floor"
            and _norm_token(getattr(clue, "value", "")) in BASEMENT_VALUES)


def basement_match(building: Building, polarity: bool = True) -> EvidenceMatch:
    """Match for "I'm in the basement": 1.0 on the below-ground floors, 0.0 elsewhere (the
    other way round for a denial). Unmatched (1.0 everywhere) in a building without one."""
    room_ids = [r.id for r in building.rooms]
    floors = set(basement_floors(building))
    if not floors:
        return EvidenceMatch({rid: 1.0 for rid in room_ids}, unmatched=True)
    on, off = (1.0, 0.0) if polarity else (0.0, 1.0)
    m = {r.id: (on if r.floor in floors else off) for r in building.rooms}
    return EvidenceMatch(m, False, [rid for rid in room_ids if m[rid] > VOCAB_MATCHED_ABOVE])


# ---------------------------------------------------------------------------
# Selector evaluation
# ---------------------------------------------------------------------------


def _within_walking_m(building: Building, room_id: str, radius_m: float) -> set[str]:
    """Rooms on ``room_id``'s floor within ``radius_m`` walking metres (same-floor edges only)."""
    rooms = {r.id: r for r in building.rooms}
    anchor = rooms.get(room_id)
    if anchor is None:
        return set()
    G = build_graph(building)
    H = nx.Graph()
    H.add_node(room_id)
    for u, v, d in G.edges(data=True):
        if d.get("kind") not in _SAME_FLOOR_KINDS or u not in rooms or v not in rooms:
            continue
        if rooms[u].floor != anchor.floor or rooms[v].floor != anchor.floor:
            continue
        H.add_edge(u, v, length_m=float(d.get("length_m", 0.0)))
    dist = nx.single_source_dijkstra_path_length(H, room_id, cutoff=radius_m, weight="length_m")
    return {rid for rid, d in dist.items() if d <= radius_m + 1e-9}


def _selector_is_empty(sel: RoomSelector) -> bool:
    return all(getattr(sel, name) is None for name in RoomSelector.model_fields)


def selector_hits(
    building: Building,
    selector: RoomSelector | None,
    meta: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    """Ids of rooms satisfying every set condition of ``selector`` (building order).

    An empty or missing selector matches nothing. ``meta`` overrides the sidecar metadata
    (``room id -> {"suite": ..., "tenant": ...}``); by default it is loaded for the building.
    """
    if selector is None or _selector_is_empty(selector):
        return []
    sel = selector
    if (sel.unit is not None or sel.tenant_contains is not None) and meta is None:
        meta = room_meta(building)
    meta = meta or {}
    unit_ids = set(unit_room_ids(building, sel.unit, meta)) if sel.unit is not None else None

    ids = set(sel.room_ids) if sel.room_ids is not None else None
    floors = {_norm_token(f) for f in sel.floor_labels} if sel.floor_labels is not None else None
    types = {_norm_token(t) for t in sel.room_types} if sel.room_types is not None else None
    name_sub = sel.name_contains.strip().lower() if sel.name_contains is not None else None
    tenant_sub = sel.tenant_contains.strip().lower() if sel.tenant_contains is not None else None
    facing = _norm_token(sel.window_facing) if sel.window_facing is not None else None
    near = (
        _within_walking_m(building, sel.within_m_of.room_id, sel.within_m_of.radius_m)
        if sel.within_m_of is not None else None
    )

    def ok(r: Room) -> bool:
        if ids is not None and r.id not in ids:
            return False
        if floors is not None and _norm_token(r.floor_label) not in floors:
            return False
        if types is not None and _norm_token(r.room_type) not in types:
            return False
        if name_sub is not None and name_sub not in r.name.lower():
            return False
        if unit_ids is not None and r.id not in unit_ids:
            return False
        if tenant_sub is not None:
            tenant = (meta.get(r.id) or {}).get("tenant")
            if not tenant or tenant_sub not in str(tenant).lower():
                return False
        if sel.has_windows is not None and r.has_windows != sel.has_windows:
            return False
        if facing is not None and (r.window_facing is None or _norm_token(r.window_facing) != facing):
            return False
        if sel.near_elevator is not None and r.near_elevator != sel.near_elevator:
            return False
        if sel.near_stairs is not None and r.near_stairs != sel.near_stairs:
            return False
        if near is not None and r.id not in near:
            return False
        return True

    return [r.id for r in building.rooms if ok(r)]


def evidence_match(
    building: Building,
    evidence: Evidence,
    vocab: dict | None = None,
    *,
    meta: Mapping[str, Mapping[str, Any]] | None = None,
) -> EvidenceMatch:
    """Per-room match values for one Evidence item (status is ignored here; fusion skips inactive items).

    ``meta`` (keyword only) overrides the building's sidecar room metadata, for tests and
    for callers that already hold it.
    """
    room_ids = [r.id for r in building.rooms]
    if evidence.kind == "vocab":
        if evidence.clue is None:
            return EvidenceMatch({rid: 1.0 for rid in room_ids}, unmatched=True)
        if is_basement_clue(evidence.clue):
            return basement_match(building, bool(evidence.clue.polarity))
        m = clue_match(building, evidence.clue, vocab)
        return EvidenceMatch(m, False, [rid for rid in room_ids if m.get(rid, 1.0) > VOCAB_MATCHED_ABOVE])

    hits = selector_hits(building, evidence.selector, meta)
    if not hits:
        return EvidenceMatch({rid: 1.0 for rid in room_ids}, unmatched=True)
    hit_value = 0.0 if evidence.negate else 1.0
    hit_set = set(hits)
    m = {rid: (hit_value if rid in hit_set else float(evidence.default_match)) for rid in room_ids}
    return EvidenceMatch(m, False, hits)
