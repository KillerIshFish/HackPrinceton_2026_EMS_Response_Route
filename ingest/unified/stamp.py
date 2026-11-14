"""Layer 5: a scan of a unit *type* stamped into every unit of that type.

An apartment record lists unit types ("B2: 2 bed, 2 bath, 95 m2"), the units of
each type, and sometimes a Matterport scan of one of them. That one scan's room
layout is a good description of every unit of the type, so its rooms and walls
are fitted into each such unit's slot:

* buildability is decided by Track A's ``preflight()`` (the ``Edge.thickness``
  check), never by whether a mesh exists;
* a buildable scan becomes a ``Model3D`` and its lowest floor is stamped;
* an unbuildable scan, a missing scan or no network leaves the unit with the
  generic bed / bath layout (the caller returns None for it).

The fit is an affine map of the scan's bounding box onto the unit's rectangle,
after turning the scan in quarter turns so its entrance side meets the corridor.
Rooms keep their order and adjacency; their proportions follow the slot.

Payloads and preflight verdicts are cached under ``data/raw/matterport/mp3d/`` so a
scan is fetched once. Given the same cached payloads the stamped geometry is
byte-identical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ingest.mp3d.model3d import Model3D, model_from_payload
from ingest.unified.lift import room_type_for
from ingest.unified.schema import Opening, Space, make_space, r

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "raw" / "matterport" / "mp3d"
ENTRY_CLASSES = ("entrance", "foyer", "hall")


def load_unit_model(model_id: str, offline: bool = False, client=None) -> tuple[Optional[Model3D], str]:
    """``(Model3D or None, status)``; status is cached | fetched | unbuildable | offline | error: ..."""
    payload_path = CACHE_DIR / f"{model_id}.payload.json"
    verdict_path = CACHE_DIR / f"{model_id}.preflight.json"
    try:
        if payload_path.is_file():
            return model_from_payload(json.loads(payload_path.read_text(encoding="utf-8")), model_id), "cached"
        if verdict_path.is_file() and not json.loads(verdict_path.read_text(encoding="utf-8")).get("buildable"):
            return None, "unbuildable"
        if offline:
            return None, "offline"
        from ingest.mp3d.client import MatterportClient
        from ingest.mp3d.fetch import fetch_geometry, save_payload
        from ingest.mp3d.preflight import preflight

        client = client or MatterportClient()
        capability = preflight(model_id, client)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        verdict_path.write_text(json.dumps({"buildable": capability.buildable, "tier": capability.tier,
                                            "reasons": list(capability.reasons)}), encoding="utf-8")
        if not capability.buildable:
            return None, "unbuildable"
        payload = fetch_geometry(model_id, client)
        save_payload(payload, payload_path)
        return model_from_payload(payload, model_id), "fetched"
    except Exception as exc:  # noqa: BLE001 - any failure degrades to the generic layout
        return None, f"error: {str(exc)[:120]}"


def _entrance_side(model: Model3D, floor: int, bbox: tuple) -> Optional[int]:
    """Side of the bounding box (0 = low y, 1 = high x, 2 = high y, 3 = low x) the entrance is on."""
    x0, y0, x1, y1 = bbox
    cls = {rm.id: (rm.classification or "").lower() for rm in model.rooms}
    best = None
    for w in sorted(model.walls, key=lambda w: w.id):
        if w.floor != floor or not w.exterior or not any(o.kind == "doorway" for o in w.openings):
            continue
        mx, my = (w.a[0] + w.b[0]) / 2, (w.a[1] + w.b[1]) / 2
        side = min(range(4), key=lambda s: (abs(my - y0), abs(mx - x1), abs(my - y1), abs(mx - x0))[s])
        entry = any(any(k in cls.get(rid, "") for k in ENTRY_CLASSES) for rid in w.room_ids)
        rank = (not entry, w.id)
        if best is None or rank < best[0]:
            best = (rank, side)
    return best[1] if best else None


def make_interior(models: dict[str, Model3D]):
    """An ``interior`` callback for ``layout.build_layout`` that stamps the given scans."""

    def interior(slot: dict, box: tuple, corridor_side: str, level: int, z0: float, height: float, walls) -> Optional[list[Space]]:
        model = next((models[s] for s in slot.get("scans") or [] if s in models), None)
        if model is None or not model.rooms:
            return None
        floor = min(rm.floor for rm in model.rooms)
        rooms = [rm for rm in model.rooms if rm.floor == floor]
        slab = model.slab(floor)
        if not rooms or slab is None:
            return None
        xs = [p[0] for rm in rooms for p in rm.polygon]
        ys = [p[1] for rm in rooms for p in rm.polygon]
        bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
        if bx1 - bx0 < 1.0 or by1 - by0 < 1.0:
            return None
        sx0, sy0, sx1, sy1 = box
        side = _entrance_side(model, floor, (bx0, by0, bx1, by1))
        target = 2 if corridor_side == "N" else 0
        if side is not None:
            k = (target - side) % 4
        else:  # no entrance found: just match long side to long side
            k = 0 if ((bx1 - bx0) >= (by1 - by0)) == ((sx1 - sx0) >= (sy1 - sy0)) else 1
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        w, d = ((bx1 - bx0), (by1 - by0)) if k % 2 == 0 else ((by1 - by0), (bx1 - bx0))

        def place(p) -> tuple[float, float]:
            dx, dy = p[0] - cx, p[1] - cy
            for _ in range(k):  # quarter turns counter-clockwise
                dx, dy = -dy, dx
            return (sx0 + (dx / w + 0.5) * (sx1 - sx0), sy0 + (dy / d + 0.5) * (sy1 - sy0))

        out: list[Space] = []
        child_id = {}
        for rm in sorted(rooms, key=lambda rm: rm.id):
            sid = f"{slot['id']}/{rm.id}"
            child_id[rm.id] = sid
            out.append(make_space(sid, level, "room", rm.classification or "Room", [place(p) for p in rm.polygon],
                                  "stamped", parent=slot["id"], raw_class=rm.classification or "",
                                  room_type=room_type_for(rm.classification), block=slot.get("_block", ""),
                                  source_ref=f"{model.model_id}:{rm.id}"))
        top = z0 + height
        for wl in sorted(model.walls, key=lambda w: w.id):
            if wl.floor != floor or wl.open_boundary:
                continue
            owner = next((child_id[i] for i in wl.room_ids if i in child_id), slot["id"])
            ops = []
            for o in wl.openings:
                oz0 = min(max(o.z0 - slab.z0 + z0, z0), top)
                oz1 = min(max(o.z1 - slab.z0 + z0, z0), top)
                if oz1 - oz0 > 0.05:
                    ops.append(Opening(o.kind, o.t0, o.t1, r(oz0), r(oz1)))
            walls.add(level, place(wl.a), place(wl.b), owner, "facade" if wl.exterior else "partition", ops,
                      thickness=r(wl.thickness_m), fidelity="stamped")
        return out

    return interior
