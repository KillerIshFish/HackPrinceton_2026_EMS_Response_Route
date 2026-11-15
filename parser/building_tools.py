"""Building look-up tools for the interpreter agent (parser/agent.py).

Pure functions over a ``Building`` and its sidecar metadata
(``ingest.real_building.load_meta(building.id)["rooms"][room_id]``, keys ``suite`` and
``tenant``). Every tool returns a compact JSON-able dict; the agent serialises it as the
``tool_result`` content.

Look-up tools (answer the model):

* ``search_rooms(query, floor_label?, room_type?, limit)``
* ``rooms_near(room_id, radius_m)``    walking distance on ``core.building_graph``, same floor
* ``floor_summary(floor_label)``

Record tools (``record_location``, ``record_vocab_clue``, ``record_fact``,
``suggest_incident_type``, ``supersede``) are defined here (schemas in ``TOOLS``) but only
acknowledged during the loop: ``record_ack`` tells the model how many rooms a selector
matches so it can correct itself. Whether a record is kept is decided in code after the
loop (``parser.agent``), never by the model.

``select_rooms`` (``core.evidence.selector_hits``) is the selector matcher used both for the
acknowledgement and for the "selector matches no room" check.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from functools import cached_property
from typing import Any, Iterable, Mapping, Optional

import networkx as nx

from core.evidence import normalize_unit, room_meta, selector_hits
from core.schema import INCIDENT_TYPES, Building, Room, RoomSelector

SEARCH_DEFAULT_LIMIT = 8
SEARCH_MAX_LIMIT = 20
NEAR_MAX_ROOMS = 20

_STOPWORDS = {
    "the", "a", "an", "of", "in", "at", "on", "to", "and", "or", "my", "our", "his", "her",
    "their", "s", "dr", "doctor", "mr", "mrs", "ms", "office", "offices", "room", "place",
    "suite", "unit", "apt", "apartment", "number", "no", "floor", "level", "inc", "llc",
}


def _tokens(text: str) -> list[str]:
    text = (text or "").lower().replace("’", "'")
    text = re.sub(r"'s\b", "", text)
    return [t for t in re.split(r"[^a-z0-9]+", text) if t]


class BuildingIndex:
    """A building plus its sidecar room metadata, with lazily built look-up structures."""

    def __init__(self, building: Building, meta: Optional[Mapping[str, Mapping[str, Any]]] = None,
                 vocab: Optional[dict] = None):
        """``meta``: ``room id -> {"suite", "tenant", ...}`` (default ``core.evidence.room_meta``).
        ``vocab``: clue vocabulary, so a search for "front desk" finds reception rooms."""
        self.building = building
        self.vocab = vocab
        self.meta_rooms: dict[str, dict] = {str(k): dict(v or {}) for k, v in
                                            (room_meta(building) if meta is None else meta).items()}
        self.rooms: dict[str, Room] = {r.id: r for r in building.rooms}

    # -- metadata ---------------------------------------------------------------------
    def suite(self, room_id: str) -> Optional[str]:
        v = (self.meta_rooms.get(room_id) or {}).get("suite")
        return str(v) if v not in (None, "") else None

    def tenant(self, room_id: str) -> Optional[str]:
        v = (self.meta_rooms.get(room_id) or {}).get("tenant")
        return str(v) if v not in (None, "") else None

    @cached_property
    def floor_labels(self) -> list[str]:
        seen: dict[str, int] = {}
        for r in self.building.rooms:
            seen.setdefault(r.floor_label, r.floor)
        return sorted(seen, key=lambda lbl: seen[lbl])

    def query_room_types(self, query: str) -> set[str]:
        """Room types the query names through the vocabulary's synonyms ("front desk" -> lobby, reception)."""
        if not self.vocab:
            return set()
        from parser.extract import extract_clues_rules

        types: set[str] = set()
        sections = self.vocab.get("room_types") or {}
        for clue in extract_clues_rules(query, self.vocab):
            if clue.type == "room_type":
                types.add(clue.value)
                types.update((sections.get(clue.value) or {}).get("also") or [])
        return types

    # -- graph ------------------------------------------------------------------------
    @cached_property
    def _graph(self) -> nx.Graph:
        from core.building_graph import build_graph

        return build_graph(self.building)

    def walking_m_from(self, room_id: str) -> dict[str, float]:
        """Walking distance (m) from ``room_id`` to every room on the same floor."""
        G = self._graph
        if room_id not in G:
            return {}
        floor = self.rooms[room_id].floor
        nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "room" and d.get("floor") == floor]
        return dict(nx.single_source_dijkstra_path_length(G.subgraph(nodes), room_id, weight="length_m"))

    # -- compact views ----------------------------------------------------------------
    def room_card(self, room_id: str) -> dict:
        r = self.rooms[room_id]
        card: dict[str, Any] = {"id": r.id, "name": r.name, "floor": r.floor_label, "type": r.room_type}
        if self.tenant(r.id):
            card["tenant"] = self.tenant(r.id)
        if self.suite(r.id):
            card["suite"] = self.suite(r.id)
        card["windows"] = r.window_facing or bool(r.has_windows)
        if r.near_stairs:
            card["near_stairs"] = True
        if r.near_elevator:
            card["near_elevator"] = True
        return card


# ---------------------------------------------------------------------------
# Selector matching
# ---------------------------------------------------------------------------


def unknown_room_ids(idx: BuildingIndex, selector: RoomSelector) -> list[str]:
    """Room ids named by the selector that do not exist in the building."""
    named = list(selector.room_ids or [])
    if selector.within_m_of is not None:
        named.append(selector.within_m_of.room_id)
    return [rid for rid in named if rid not in idx.rooms]


def selector_is_empty(selector: RoomSelector) -> bool:
    return not selector.model_dump(exclude_none=True)


def select_rooms(idx: BuildingIndex, selector: RoomSelector) -> list[str]:
    """Ids of the rooms satisfying every set condition of ``selector`` (building order).

    Delegates to ``core.evidence.selector_hits`` so the agent's "matches no room" check and
    the posterior's match values can never disagree.
    """
    return selector_hits(idx.building, selector, idx.meta_rooms)


# ---------------------------------------------------------------------------
# Look-up tools
# ---------------------------------------------------------------------------


def search_rooms(
    idx: BuildingIndex,
    query: str,
    floor_label: Optional[str] = None,
    room_type: Optional[str] = None,
    limit: int = SEARCH_DEFAULT_LIMIT,
) -> dict:
    """Rooms whose name, tenant, suite / unit number, type or id match words of ``query``."""
    limit = max(1, min(int(limit or SEARCH_DEFAULT_LIMIT), SEARCH_MAX_LIMIT))
    words = [t for t in _tokens(query) if t not in _STOPWORDS] or _tokens(query)
    q_unit = normalize_unit(query)
    q_types = idx.query_room_types(query)
    scored: list[tuple[float, str]] = []
    for r in idx.building.rooms:
        if floor_label is not None and r.floor_label != str(floor_label):
            continue
        if room_type is not None and r.room_type.lower() != room_type.lower():
            continue
        suite = idx.suite(r.id)
        tenant = idx.tenant(r.id) or ""
        hay = set(_tokens(r.name)) | set(_tokens(tenant)) | set(_tokens(r.room_type.replace("_", " "))) | set(_tokens(r.id))
        score = 0.0
        if suite and q_unit and normalize_unit(suite) == q_unit:
            score += 3.0
        for w in words:
            if suite and w == normalize_unit(suite):
                score += 2.0
            elif w in hay:
                score += 1.0
            elif len(w) >= 4 and any(h.startswith(w) or w.startswith(h) for h in hay if len(h) >= 4):
                score += 0.5  # "therapy" ~ "therapist", speech-recognition fragments
        if r.room_type in q_types:
            score += 1.5
        if not words and (floor_label is not None or room_type is not None):
            score = 1.0  # a pure filter query
        if score > 0:
            scored.append((score, r.id))
    scored.sort(key=lambda s: -s[0])
    return {"query": query, "total": len(scored), "rooms": [idx.room_card(rid) for _, rid in scored[:limit]]}


def rooms_near(idx: BuildingIndex, room_id: str, radius_m: float) -> dict:
    """Rooms within ``radius_m`` walking metres of ``room_id`` on the same floor."""
    if room_id not in idx.rooms:
        return {"error": f"unknown room id '{room_id}'"}
    dist = idx.walking_m_from(room_id)
    hits = sorted(((d, rid) for rid, d in dist.items() if rid != room_id and d <= float(radius_m)))
    rooms = [dict(idx.room_card(rid), walk_m=round(d, 1)) for d, rid in hits[:NEAR_MAX_ROOMS]]
    return {"room_id": room_id, "radius_m": radius_m, "total": len(hits), "rooms": rooms}


def floor_summary(idx: BuildingIndex, floor_label: str) -> dict:
    """Count by room type, tenants with suites, which sides have windows."""
    rooms = [r for r in idx.building.rooms if r.floor_label == str(floor_label)]
    if not rooms:
        return {"error": f"no floor labelled '{floor_label}'", "floors": idx.floor_labels}
    tenants: dict[str, set[str]] = defaultdict(set)
    for r in rooms:
        suite = idx.suite(r.id)
        if suite or idx.tenant(r.id):
            status = (idx.meta_rooms.get(r.id) or {}).get("status")
            tenants[suite or "?"].add(idx.tenant(r.id) or ("vacant" if status == "available" else "tenant not on record"))
    return {
        "floor": str(floor_label),
        "rooms": len(rooms),
        "by_type": dict(Counter(r.room_type for r in rooms).most_common()),
        "suites": {s: sorted(t) for s, t in sorted(tenants.items())},
        "window_sides": sorted({r.window_facing for r in rooms if r.window_facing}),
        "rooms_with_windows": sum(1 for r in rooms if r.has_windows),
    }


def building_digest(idx: BuildingIndex) -> str:
    """Compact text description of the building for the cached system block."""
    b = idx.building
    lines = [f"Building id: {b.id}", f"Name: {b.name}", f"Use: {b.use_type}",
             f"Floor labels (bottom to top): {', '.join(idx.floor_labels)}", ""]
    for label in idx.floor_labels:
        s = floor_summary(idx, label)
        types = ", ".join(f"{n} {t}" for t, n in s["by_type"].items())
        lines.append(f"Floor {label}: {s['rooms']} rooms ({types})")
        if s["window_sides"]:
            lines.append(f"  window sides: {', '.join(s['window_sides'])}")
        for suite, tenants in s["suites"].items():
            lines.append(f"  suite {suite}: {'; '.join(tenants)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Room list for the cached system block
# ---------------------------------------------------------------------------

# Up to this many rooms the whole building is listed, so the model can place a caller in one
# request without a look-up. Larger buildings list their labelled rooms only (numbers,
# tenants) and the model uses the look-up tools for the rest.
ROSTER_MAX_ROOMS = 1500
# Neighbours ("next to", "across") are listed up to this building size.
ROSTER_NEIGHBOUR_MAX_ROOMS = 320
_CIRCULATION = {"corridor", "stairwell", "elevator"}
_WALL_GAP_M = 0.6  # two rooms share a wall when their outlines come this close
_ACROSS_MAX_M = 16.0


def _point_segment_m(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay, bx, by, px, py = a[0], a[1], b[0], b[1], p[0], p[1]
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    t = 0.0 if den <= 0.0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / den))
    cx, cy = ax + t * dx, ay + t * dy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def _outline_gap_m(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Smallest distance between the outlines of two rooms (0 when they touch)."""
    if len(a) < 2 or len(b) < 2:
        return float("inf")
    best = float("inf")
    for i in range(len(a)):
        a1, a2 = a[i], a[(i + 1) % len(a)]
        for j in range(len(b)):
            b1, b2 = b[j], b[(j + 1) % len(b)]
            best = min(best, _point_segment_m(a1, b1, b2), _point_segment_m(a2, b1, b2),
                       _point_segment_m(b1, a1, a2), _point_segment_m(b2, a1, a2))
            if best <= 1e-9:
                return 0.0
    return best


def room_neighbours(idx: BuildingIndex) -> dict[str, dict[str, list[str]]]:
    """``room id -> {"next_to": [...], "across": [...]}`` from the floor geometry.

    ``next_to``: rooms sharing a wall (corridors, stairs and elevators are not listed as
    neighbours of a room; a corridor lists the rooms that open off it). ``across``: rooms on
    the other side of a corridor this room touches, nearest first. Same floor only.
    """
    by_floor: dict[int, list[Room]] = defaultdict(list)
    for r in idx.building.rooms:
        by_floor[r.floor].append(r)
    out: dict[str, dict[str, list[str]]] = {}
    for rooms in by_floor.values():
        boxes = {r.id: (min(p[0] for p in r.polygon), min(p[1] for p in r.polygon),
                        max(p[0] for p in r.polygon), max(p[1] for p in r.polygon))
                 for r in rooms if r.polygon}
        touching: dict[str, set[str]] = defaultdict(set)
        for i, a in enumerate(rooms):
            for b in rooms[i + 1:]:
                if a.id not in boxes or b.id not in boxes:
                    continue
                ba, bb = boxes[a.id], boxes[b.id]
                if (ba[0] - bb[2] > _WALL_GAP_M or bb[0] - ba[2] > _WALL_GAP_M
                        or ba[1] - bb[3] > _WALL_GAP_M or bb[1] - ba[3] > _WALL_GAP_M):
                    continue
                if _outline_gap_m(list(a.polygon), list(b.polygon)) <= _WALL_GAP_M:
                    touching[a.id].add(b.id)
                    touching[b.id].add(a.id)
        by_id = {r.id: r for r in rooms}

        def dist(a: str, b: str) -> float:
            ca, cb = by_id[a].centroid, by_id[b].centroid
            return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5

        for r in rooms:
            walls = sorted((o for o in touching[r.id] if by_id[o].room_type not in _CIRCULATION),
                           key=lambda o: (round(dist(r.id, o), 3), o))  # id breaks ties: the text must not vary between runs
            across: list[str] = []
            if r.room_type not in _CIRCULATION:
                seen = set(touching[r.id]) | {r.id}
                cands: set[str] = set()
                for c in touching[r.id]:
                    if by_id[c].room_type == "corridor":
                        cands |= {o for o in touching[c] if o not in seen and by_id[o].room_type not in _CIRCULATION}
                across = sorted((o for o in cands if dist(r.id, o) <= _ACROSS_MAX_M),
                                key=lambda o: (round(dist(r.id, o), 3), o))[:3]
            limit = 8 if r.room_type in _CIRCULATION else 4
            out[r.id] = {"next_to": walls[:limit], "across": across}
    return out


def roster_line(idx: BuildingIndex, room: Room, neighbours: Optional[dict[str, dict[str, list[str]]]] = None) -> str:
    """One room for the system block: ``id | name | type, area | what is known about it``."""
    parts = [room.id, room.name, f"{room.room_type}, {round(room.area_m2)} m2"]
    facts: list[str] = []
    suite, tenant = idx.suite(room.id), idx.tenant(room.id)
    if suite:
        facts.append(f"suite {suite}")
    if tenant:
        facts.append(f"tenant {tenant}")
    elif (idx.meta_rooms.get(room.id) or {}).get("status") == "available":
        facts.append("vacant")
    if room.window_facing:
        facts.append(f"windows face {room.window_facing}")
    elif room.has_windows:
        facts.append("has windows")
    if room.near_stairs:
        facts.append("by stairs")
    if room.near_elevator:
        facts.append("by elevator")
    if neighbours is not None:
        n = neighbours.get(room.id) or {}
        if n.get("next_to"):
            facts.append("next to: " + ", ".join(n["next_to"]))
        if n.get("across"):
            facts.append("across the corridor: " + ", ".join(n["across"]))
    if facts:
        parts.append("; ".join(facts))
    return " | ".join(parts)


def building_roster(idx: BuildingIndex) -> str:
    """The building for the cached system block: every room, floor by floor.

    One line per room (``roster_line``). Buildings over ``ROSTER_MAX_ROOMS`` rooms list only
    the rooms that carry a number or a tenant, and say so; the look-up tools cover the rest.
    """
    b = idx.building
    rooms = list(b.rooms)
    partial = len(rooms) > ROSTER_MAX_ROOMS
    if partial:
        from core.evidence import room_labels

        labelled = room_labels(b, idx.meta_rooms)
        rooms = [r for r in rooms if r.id in labelled or idx.tenant(r.id)][:ROSTER_MAX_ROOMS]
    neighbours = room_neighbours(idx) if len(b.rooms) <= ROSTER_NEIGHBOUR_MAX_ROOMS else None
    lines = [f"Building id: {b.id}", f"Name: {b.name}", f"Use: {b.use_type}",
             f"Floor labels, bottom to top: {', '.join(idx.floor_labels)}",
             "Each line: room id | name | type, floor area | what is known about it."]
    if partial:
        lines.append(f"This building has {len(b.rooms)} rooms; only the {len(rooms)} with a number or a tenant are "
                     "listed. Use search_rooms and floor_summary for the others.")
    if neighbours is None:
        lines.append("Neighbours are not listed for a building this size; use rooms_near for 'next to' and 'across from'.")
    listed = {r.id for r in rooms}
    for label in idx.floor_labels:
        on_floor = [r for r in b.rooms if r.floor_label == label and r.id in listed]
        if not on_floor:
            continue
        lines += ["", f"Floor {label} ({len(on_floor)} rooms{' listed' if partial else ''})"]
        lines.extend(roster_line(idx, r, neighbours) for r in on_floor)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool schemas (strict). Optional properties are simply left out of `required`.
# ---------------------------------------------------------------------------

CLUE_TYPES = ["room_type", "floor", "landmark", "view", "sound", "smell", "negative", "route"]
FACT_CATEGORIES = ["hazard", "access", "patient", "people", "threat", "resource"]
PREMISE_KEYS = ["means_of_entry", "meet_person", "elevator", "defibrillator"]

_SPAN = {"type": "string", "description": "The caller's exact words, copied character for character from a caller turn."}
_CONF = {"type": "number", "description": "0 to 1. How sure the statement places / describes what you record."}
_CERTAINTY = {"type": "number",
              "description": "0 to 1. How sure the CALLER sounds about this statement: 1.0 said plainly, about 0.5 "
                             "one hedge or a questioning tone, 0.2 or less when they are guessing."}
_SPECIFICITY = {"type": "number",
                "description": "0 to 1. How exactly the caller's words pin the place down: 1.0 a number or name "
                               "given in full, about 0.5 approximate, partial or garbled, lower when barely narrowed."}

_SELECTOR_SCHEMA = {
    "type": "object",
    "description": "Rooms the statement points at. Usually just room_ids. The other conditions are for statements "
                   "too broad to list room by room; set only what the caller's words support, they AND together.",
    "properties": {
        "room_ids": {"type": "array", "items": {"type": "string"},
                     "description": "Room ids copied exactly from the room list (or a tool result): every room the "
                                    "caller's words point at."},
        "floor_labels": {"type": "array", "items": {"type": "string"}},
        "room_types": {"type": "array", "items": {"type": "string"}},
        "name_contains": {"type": "string"},
        "unit": {"type": "string", "description": "A suite, unit or room number as printed on the room, e.g. '304'."},
        "tenant_contains": {"type": "string", "description": "Part of the tenant name, e.g. 'Patel'."},
        "has_windows": {"type": "boolean"},
        "window_facing": {"type": "string"},
        "near_elevator": {"type": "boolean"},
        "near_stairs": {"type": "boolean"},
        "within_m_of": {
            "type": "object",
            "properties": {"room_id": {"type": "string"}, "radius_m": {"type": "number"}},
            "required": ["room_id", "radius_m"],
            "additionalProperties": False,
        },
    },
    "required": [],
    "additionalProperties": False,
}


def _tool(name: str, description: str, properties: dict, required: Iterable[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(required),
            "additionalProperties": False,
        },
    }


TOOLS: list[dict] = [
    _tool(
        "search_rooms",
        "Look-up, only needed when the room list in the briefing is partial. Finds rooms by words the caller "
        "used: a tenant or business name, a suite, unit or room number, a room name or a room type. Returns id, "
        "name, floor, type, tenant, suite, windows (side they face, or true/false), near_stairs, near_elevator.",
        {
            "query": {"type": "string"},
            "floor_label": {"type": "string", "description": "Only rooms on this floor label."},
            "room_type": {"type": "string", "description": "Only rooms of this type."},
            "limit": {"type": "integer", "description": "Max rooms returned (default 8, max 20)."},
        },
        ["query"],
    ),
    _tool(
        "rooms_near",
        "Look-up: rooms within radius_m walking metres of a room, same floor. For 'down the hall from', 'a few "
        "doors past', or 'next to' when the room list shows no neighbours.",
        {"room_id": {"type": "string"}, "radius_m": {"type": "number"}},
        ["room_id", "radius_m"],
    ),
    _tool(
        "floor_summary",
        "Look-up, only needed when the room list in the briefing is partial. One floor: room count by type, "
        "suites and tenants, which sides have windows.",
        {"floor_label": {"type": "string"}},
        ["floor_label"],
    ),
    _tool(
        "record_location",
        "Record where the caller is: the rooms their words point at. confidence is your own reading: the chance "
        "these are the rooms the words point at. certainty and specificity rate the caller's statement and set "
        "how much it weighs. For a denial ('we're not in 304') set negate=true and select the rooms they ruled "
        "out; otherwise negate=false.",
        {
            "selector": _SELECTOR_SCHEMA,
            "negate": {"type": "boolean"},
            "confidence": _CONF,
            "certainty": _CERTAINTY,
            "specificity": _SPECIFICITY,
            "span": _SPAN,
            "rationale": {"type": "string", "description": "A few words: why these rooms."},
        },
        ["selector", "negate", "confidence", "certainty", "specificity", "span", "rationale"],
    ),
    _tool(
        "record_vocab_clue",
        "Record a clue from the controlled vocabulary in the briefing (type + key): a floor, a view, a sound, a "
        "smell, a landmark, a route, or a room type that many rooms share. It narrows the picture without naming "
        "a room. For a denial use type 'negative' with value '<section>:<key>' (e.g. 'views:parking_lot') and "
        "polarity false. For type 'floor' also give certainty and specificity (the caller's statement of their "
        "floor is weighed by them); leave both out for every other type.",
        {
            "type": {"type": "string", "enum": CLUE_TYPES},
            "value": {"type": "string"},
            "polarity": {"type": "boolean"},
            "confidence": _CONF,
            "certainty": _CERTAINTY,
            "specificity": _SPECIFICITY,
            "span": _SPAN,
        },
        ["type", "value", "polarity", "confidence", "span"],
    ),
    _tool(
        "rate_statement",
        "Rate a location statement the fixed rules already read from the newest caller turn. The message lists "
        "each with a label (s1, s2, ...). The rules matched the number or name; you say how sure the caller "
        "sounded and how exact the words were. One call per label.",
        {
            "statement": {"type": "string", "description": "The label from the message, e.g. 's1'."},
            "certainty": _CERTAINTY,
            "specificity": _SPECIFICITY,
        },
        ["statement", "certainty", "specificity"],
    ),
    _tool(
        "record_fact",
        "Record something that matters to the response but is not the caller's location: hazard, access, "
        "patient, people, threat or resource. Write text as what the caller reported ('Caller reports ...'), "
        "never as verified.",
        {
            "category": {"type": "string", "enum": FACT_CATEGORIES},
            "text": {"type": "string"},
            "span": _SPAN,
            "room_id": {"type": "string", "description": "Only if the caller tied it to one room: that room's id from the room list."},
            "premise_key": {"type": "string", "enum": PREMISE_KEYS,
                            "description": "Only if it answers one of these: means_of_entry, meet_person, elevator, defibrillator."},
        },
        ["category", "text", "span"],
    ),
    _tool(
        "suggest_incident_type",
        "Suggest what kind of emergency this is. The dispatcher decides; this only offers it.",
        {"type": {"type": "string", "enum": [t for t in INCIDENT_TYPES if t != "unknown"]}, "span": _SPAN},
        ["type", "span"],
    ),
    _tool(
        "supersede",
        "The caller corrected an earlier statement: mark that earlier evidence id (from the list in the "
        "message) as replaced. Record the corrected statement separately.",
        {"evidence_id": {"type": "string"}, "span": _SPAN},
        ["evidence_id", "span"],
    ),
]

LOOKUP_TOOLS = ("search_rooms", "rooms_near", "floor_summary")
RECORD_TOOLS = ("record_location", "record_vocab_clue", "record_fact", "suggest_incident_type", "supersede")
# Rates a statement the dialog rules read (no span of its own, nothing to look up).
RATE_TOOLS = ("rate_statement",)


def run_lookup(idx: BuildingIndex, name: str, args: dict) -> dict:
    """Execute a look-up tool. Unknown names and bad arguments come back as ``{"error": ...}``."""
    try:
        if name == "search_rooms":
            return search_rooms(idx, str(args.get("query", "")), args.get("floor_label"),
                                args.get("room_type"), args.get("limit") or SEARCH_DEFAULT_LIMIT)
        if name == "rooms_near":
            return rooms_near(idx, str(args.get("room_id", "")), float(args.get("radius_m", 0.0)))
        if name == "floor_summary":
            return floor_summary(idx, str(args.get("floor_label", "")))
    except (TypeError, ValueError) as exc:
        return {"error": f"bad arguments: {exc}"}
    return {"error": f"unknown tool '{name}'"}


def record_ack(idx: BuildingIndex, name: str, args: dict, record_no: int, turn_text: Optional[str] = None,
               earlier_texts: Iterable[str] = ()) -> dict:
    """Acknowledge a record call. Final keep / drop happens in code after the loop.

    A ``warning`` (or ``error``) says the record will be dropped as it stands: its span is not
    in the newest caller turn (``turn_text``, when given), a room id does not exist, or the
    selector matches no room. The loop sends the acknowledgements back only when one carries
    a warning, so the model can correct that record; otherwise one request is enough.
    """
    ack: dict[str, Any] = {"noted": f"r{record_no}", "note": "checked against the transcript and building after you finish"}
    if turn_text is not None:
        def flat(text: str) -> str:
            return " ".join(text.split()).lower().replace("’", "'")

        span = flat(str(args.get("span", "")))
        if not span or span not in flat(turn_text):
            if span and any(span in flat(t) for t in earlier_texts):
                ack["note"] = "these words are from an earlier turn, which was read then: not recorded again"
            else:
                ack["warning"] = ("span is not word for word in the newest caller turn: this record will be "
                                  "dropped. Send it again with the caller's exact words if it matters.")
            return ack
    fact_room = str(args.get("room_id") or "").strip()
    if name == "record_fact" and fact_room.lower() not in ("", "null", "none") and fact_room not in idx.rooms:
        ack["warning"] = f"unknown room id '{fact_room}': this fact will be dropped"
        return ack
    if name == "record_location":
        try:
            sel = RoomSelector.model_validate(args.get("selector") or {})
        except (ValueError, TypeError) as exc:
            return {"error": f"invalid selector: {exc.errors()[0]['msg'] if hasattr(exc, 'errors') else exc}"}
        bad = unknown_room_ids(idx, sel)
        if bad:
            ack["warning"] = f"unknown room ids {bad}: this record will be dropped"
            return ack
        hits = select_rooms(idx, sel)
        ack["matches"] = len(hits)
        ack["sample"] = hits[:6]
        if not hits:
            ack["warning"] = "selector matches no room: this record will be dropped"
    return ack
