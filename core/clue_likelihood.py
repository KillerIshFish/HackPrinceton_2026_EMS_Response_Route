"""Clue -> match vector m_i(r) in [0, 1] over rooms.

Driven entirely by configs/clue_vocab.yaml:

* room_type: m = 1 if Room.room_type == value or is listed in that entry's `also`.
  A composite room (`composite_room_types`, e.g. a whole unscanned tenant suite)
  gets the partial score listed under its `contains` for the interior room type.
* view / landmark / sound / smell: m = 1 if the room satisfies the entry's
  `match` dict (ALL keys hold; a list value means any-of), else `miss` (default 0).
* floor: m = 1 on the stated floor, `adjacent_floor_match` on floors +/-1, else 0.
  Value is a floor label ("4"), an ordinal word, or "top" / "ground".
* negative: value is "<section>:<key>" or a bare key; always scored as 1 - m.
* route: value is a key of `routes` ("stairs_up_right"); m comes from the turns on the
  walk from where that anchor arrives to each room (core/route_clues.py).

Polarity is applied here (polarity False -> 1 - m). Unknown values are
uninformative: every room gets 1.0.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from core.route_clues import route_match
from core.schema import Building, Clue, Room

VOCAB_PATH = Path(__file__).resolve().parents[1] / "configs" / "clue_vocab.yaml"

# Clue.type -> vocab section
TYPE_TO_SECTION = {
    "room_type": "room_types",
    "view": "views",
    "landmark": "landmarks",
    "sound": "sounds",
    "smell": "smells",
    "floor": "floors",
    "route": "routes",
}
# Order used to resolve a bare key (negative clues, or a key filed under another section).
SECTION_SEARCH_ORDER = ("room_types", "views", "landmarks", "sounds", "smells", "routes")
_SECTION_ALIASES = {
    **{s: s for s in SECTION_SEARCH_ORDER},
    **TYPE_TO_SECTION,
    "floors": "floors",
}


@lru_cache(maxsize=8)
def _load_vocab_cached(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_vocab(path: str | Path | None = None) -> dict:
    """Load configs/clue_vocab.yaml (or ``path``) as a dict."""
    return _load_vocab_cached(str(Path(path) if path is not None else VOCAB_PATH))


def _norm(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _contains_weight(room: Room, wanted_types, vocab: dict) -> float:
    """How well a composite room (a whole unscanned suite) matches interior room types.

    ``composite_room_types.<type>.contains`` in the vocab says how likely such a
    room is to contain each interior room type. 0.0 for ordinary rooms.
    """
    contains = ((vocab.get("composite_room_types") or {}).get(room.room_type) or {}).get("contains") or {}
    return max((float(contains.get(t, 0.0)) for t in wanted_types), default=0.0)


def _satisfies(room: Room, match: dict, vocab: dict | None = None) -> float:
    """1.0 when the room meets every condition (list value = any-of), else 0.0.

    A ``room_type`` condition is met partially by a composite room that probably
    contains a room of that type (see ``_contains_weight``).
    """
    score = 1.0
    for field, wanted in match.items():
        actual = getattr(room, field, None)
        wanted_list = list(wanted) if isinstance(wanted, (list, tuple, set)) else [wanted]
        if actual in wanted_list:
            continue
        if field == "room_type" and vocab is not None:
            score = min(score, _contains_weight(room, wanted_list, vocab))
            if score > 0.0:
                continue
        return 0.0
    return score


def _room_type_match(building: Building, key: str, entry: dict | None, vocab: dict | None = None) -> dict[str, float]:
    accepted = {key, *((entry or {}).get("also") or [])}
    out: dict[str, float] = {}
    for r in building.rooms:
        if r.room_type in accepted:
            out[r.id] = 1.0
        else:
            out[r.id] = _contains_weight(r, accepted, vocab) if vocab is not None else 0.0
    return out


def _rule_match(building: Building, entry: dict, vocab: dict | None = None) -> dict[str, float] | None:
    match = entry.get("match")
    if not match:
        return None
    miss = float(entry.get("miss", 0.0))
    return {r.id: max(miss, _satisfies(r, match, vocab)) for r in building.rooms}


def _resolve_floor(building: Building, value: str, floors_cfg: dict) -> int | None:
    """Floor index for a label ("4"), ordinal word ("fourth"), or "top" / "ground"."""
    if not building.floors:
        return None
    v = _norm(value)
    for suffix in ("_floor", "_level"):
        if v.endswith(suffix):
            v = v[: -len(suffix)]
    if v in ("top", "highest", "roof"):
        return max(building.floors)
    if v in ("ground", "main", "street"):
        return min(building.floors, key=lambda f: (abs(building.floor_heights_m.get(f, 0.0)), f))
    for room in building.rooms:
        if _norm(room.floor_label) == v:
            return room.floor
    ordinals = {_norm(k): int(n) for k, n in (floors_cfg.get("ordinals") or {}).items()}
    if v in ordinals:
        return ordinals[v] if ordinals[v] in building.floors else None
    try:
        idx = int(v)
    except ValueError:
        return None
    return idx if idx in building.floors else None


def _floor_match(building: Building, value: str, floors_cfg: dict) -> dict[str, float] | None:
    target = _resolve_floor(building, value, floors_cfg)
    if target is None:
        return None
    adjacent = float(floors_cfg.get("adjacent_floor_match", 0.0))
    out: dict[str, float] = {}
    for room in building.rooms:
        diff = abs(room.floor - target)
        out[room.id] = 1.0 if diff == 0 else (adjacent if diff == 1 else 0.0)
    return out


def _section_match(building: Building, section: str, key: str, vocab: dict) -> dict[str, float] | None:
    """Positive match vector for `key` in `section`, or None if it is not defined there."""
    if section == "floors":
        return _floor_match(building, key, vocab.get("floors") or {})
    entries = vocab.get(section) or {}
    k = _norm(key)
    if k not in entries:
        return None
    entry = entries[k] or {}
    if section == "room_types":
        return _room_type_match(building, k, entry, vocab)
    if section == "routes":
        return route_match(building, str(entry.get("anchor", "")), str(entry.get("turn", "")))
    return _rule_match(building, entry, vocab)


def _positive_match(building: Building, clue_type: str, value: str, vocab: dict) -> dict[str, float] | None:
    """Positive (polarity-free) match vector, or None when the value is unknown."""
    value = str(value).strip()
    sections: list[str] = []

    if ":" in value:  # explicit "<section>:<key>"
        prefix, _, rest = value.partition(":")
        section = _SECTION_ALIASES.get(_norm(prefix))
        if section is not None:
            return _section_match(building, section, rest, vocab)

    own = TYPE_TO_SECTION.get(clue_type)
    if own is not None:
        sections.append(own)
    sections.extend(s for s in SECTION_SEARCH_ORDER if s not in sections)

    for section in sections:
        m = _section_match(building, section, value, vocab)
        if m is not None:
            return m
    # A room type that exists in the building but not (yet) in the vocab still matches itself.
    if clue_type in ("room_type", "negative"):
        k = _norm(value)
        if any(r.room_type == k for r in building.rooms):
            return _room_type_match(building, k, None, vocab)
    if clue_type == "negative":  # e.g. "not on the 4th floor"
        return _floor_match(building, value, vocab.get("floors") or {})
    return None


def clue_match(building: Building, clue: Clue, vocab: dict | None = None) -> dict[str, float]:
    """m_i(r) in [0, 1] for every room, polarity already applied.

    polarity False or type "negative" -> 1 - m. Unknown value -> all 1.0.
    """
    vocab = vocab if vocab is not None else load_vocab()
    m = _positive_match(building, clue.type, clue.value, vocab)
    if m is None:
        return {r.id: 1.0 for r in building.rooms}
    if clue.type == "negative" or not clue.polarity:
        return {rid: 1.0 - v for rid, v in m.items()}
    return m
