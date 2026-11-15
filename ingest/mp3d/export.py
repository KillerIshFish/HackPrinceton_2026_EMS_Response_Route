"""Exporters.

JSON is the canonical output and lives on :class:`~ingest.mp3d.model3d.Model3D`;
it keeps the room and opening semantics that a mesh format would flatten away.
This module adds glTF 2.0 for anything that just wants to open a 3D file.

Axis convention: glTF is y-up, so model ``(x, y)`` maps to ``(x, elevation, -y)``.
That puts model +y toward -z, i.e. "north into the screen" at the default camera.
"""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Sequence

from ingest.mp3d.model3d import Model3D, RoomPlate, Wall

GLTF_VERSION = "2.0"
PLATE_LIFT_M = 0.02  # keep plates off the slab plane so they do not z-fight


class _MeshBuffer:
    """Accumulates interleaved-free position/index arrays for one primitive."""

    def __init__(self) -> None:
        self.positions: list[float] = []
        self.indices: list[int] = []

    @property
    def vertex_count(self) -> int:
        return len(self.positions) // 3

    def add_triangle(
        self,
        p0: Sequence[float],
        p1: Sequence[float],
        p2: Sequence[float],
    ) -> None:
        base = self.vertex_count
        for p in (p0, p1, p2):
            self.positions.extend([float(p[0]), float(p[1]), float(p[2])])
        self.indices.extend([base, base + 1, base + 2])

    def add_quad(
        self,
        p0: Sequence[float],
        p1: Sequence[float],
        p2: Sequence[float],
        p3: Sequence[float],
    ) -> None:
        self.add_triangle(p0, p1, p2)
        self.add_triangle(p0, p2, p3)

    def bounds(self) -> tuple[list[float], list[float]]:
        if not self.positions:
            return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        xs = self.positions[0::3]
        ys = self.positions[1::3]
        zs = self.positions[2::3]
        return ([min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)])


def _wall_box(buf: _MeshBuffer, wall: Wall, t0: float, t1: float, z0: float, z1: float) -> None:
    """Add one solid wall panel as a closed box with real thickness."""
    ax, ay = wall.a
    bx, by = wall.b
    dx, dy = bx - ax, by - ay
    length = (dx * dx + dy * dy) ** 0.5
    if length <= 0:
        return
    ux, uy = dx / length, dy / length
    half = max(wall.thickness_m, 0.02) / 2.0  # floor at 2 cm so a zero-thickness edge still renders as a box
    nx, ny = -uy * half, ux * half

    sx, sy = ax + dx * t0, ay + dy * t0
    ex, ey = ax + dx * t1, ay + dy * t1

    # four footprint corners, in glTF axes (x, y=elevation, z=-model_y)
    c = [
        (sx + nx, -(sy + ny)),
        (ex + nx, -(ey + ny)),
        (ex - nx, -(ey - ny)),
        (sx - nx, -(sy - ny)),
    ]
    lo = [(p[0], z0, p[1]) for p in c]
    hi = [(p[0], z1, p[1]) for p in c]

    buf.add_quad(hi[0], hi[1], hi[2], hi[3])              # top
    buf.add_quad(lo[3], lo[2], lo[1], lo[0])              # bottom
    for i in range(4):
        j = (i + 1) % 4
        buf.add_quad(lo[i], lo[j], hi[j], hi[i])          # sides


def _plate(buf: _MeshBuffer, room: RoomPlate, z: float) -> None:
    poly = room.polygon
    for ia, ib, ic in room.triangles:
        if max(ia, ib, ic) >= len(poly):
            continue
        a, b, c = poly[ia], poly[ib], poly[ic]
        buf.add_triangle((a[0], z, -a[1]), (b[0], z, -b[1]), (c[0], z, -c[1]))


def _pack(
    buffers: Sequence[_MeshBuffer],
) -> tuple[bytes, list[dict], list[dict], list[tuple[int, int] | None]]:
    """Concatenate primitives into one glTF buffer.

    Returns the blob, bufferViews, accessors, and per-input-primitive
    ``(position_accessor, index_accessor)`` or ``None`` when that primitive had
    no geometry. Empty primitives consume no accessor slots, so indices stay
    aligned with what is actually emitted.
    """
    blob = bytearray()
    views: list[dict] = []
    accessors: list[dict] = []
    slots: list[tuple[int, int] | None] = []
    # glTF enums used below: target 34962/34963 = ARRAY_BUFFER/ELEMENT_ARRAY_BUFFER, componentType 5126 = FLOAT,
    # 5125 = UNSIGNED_INT. Each bufferView is padded to a 4-byte boundary, which glTF requires for these types.
    for buf in buffers:
        if not buf.indices:
            slots.append(None)
            continue
        while len(blob) % 4:
            blob.append(0)
        pos_offset = len(blob)
        blob.extend(struct.pack("<%df" % len(buf.positions), *buf.positions))
        pos_view = len(views)
        views.append({"buffer": 0, "byteOffset": pos_offset,
                      "byteLength": len(buf.positions) * 4, "target": 34962})

        while len(blob) % 4:
            blob.append(0)
        idx_offset = len(blob)
        blob.extend(struct.pack("<%dI" % len(buf.indices), *buf.indices))
        idx_view = len(views)
        views.append({"buffer": 0, "byteOffset": idx_offset,
                      "byteLength": len(buf.indices) * 4, "target": 34963})

        mn, mx = buf.bounds()
        pos_acc = len(accessors)
        accessors.append({
            "bufferView": pos_view, "componentType": 5126,
            "count": buf.vertex_count, "type": "VEC3", "min": mn, "max": mx,
        })
        idx_acc = len(accessors)
        accessors.append({
            "bufferView": idx_view, "componentType": 5125,
            "count": len(buf.indices), "type": "SCALAR",
        })
        slots.append((pos_acc, idx_acc))
    return bytes(blob), views, accessors, slots


def to_gltf(model: Model3D) -> dict:
    """A self-contained glTF 2.0 document (buffer embedded as a data URI).

    One node per floor, each with two primitives: translucent walls and opaque
    floor plates. Openings are already cut out, so holes are real geometry.
    """
    prims: list[tuple[str, _MeshBuffer, int, int]] = []  # name, buf, floor, material
    for slab in model.slabs:
        wall_buf = _MeshBuffer()
        for wall in model.walls:
            if wall.floor != slab.floor or wall.open_boundary:
                continue
            for panel in model.panels(wall):
                _wall_box(wall_buf, wall, panel.t0, panel.t1, panel.z0, panel.z1)
        plate_buf = _MeshBuffer()
        for room in model.rooms:
            if room.floor == slab.floor:
                _plate(plate_buf, room, slab.z0 + PLATE_LIFT_M)
        prims.append((f"floor{slab.floor}_walls", wall_buf, slab.floor, 0))
        prims.append((f"floor{slab.floor}_plates", plate_buf, slab.floor, 1))

    blob, views, accessors, slots = _pack([buf for _n, buf, _f, _m in prims])

    meshes: list[dict] = []
    nodes: list[dict] = []
    for (name, _buf, _floor, material), slot in zip(prims, slots):
        if slot is None:
            continue
        pos_acc, idx_acc = slot
        meshes.append({
            "name": name,
            "primitives": [{
                "attributes": {"POSITION": pos_acc},
                "indices": idx_acc,
                "material": material,
                "mode": 4,  # glTF TRIANGLES
            }],
        })
        nodes.append({"name": name, "mesh": len(meshes) - 1})

    doc = {
        "asset": {
            "version": GLTF_VERSION,
            "generator": f"ingest.mp3d ({model.schema}) from Matterport {model.model_id}",
        },
        "scene": 0,
        "scenes": [{"name": model.name or model.model_id,
                    "nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": [
            {
                "name": "wall",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.62, 0.68, 0.70, 0.35],
                    "metallicFactor": 0.0, "roughnessFactor": 0.9,
                },
                "alphaMode": "BLEND", "doubleSided": True,
            },
            {
                "name": "floor_plate",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.86, 0.89, 0.90, 1.0],
                    "metallicFactor": 0.0, "roughnessFactor": 0.95,
                },
                "doubleSided": True,
            },
        ],
        "buffers": [{
            "byteLength": len(blob),
            "uri": "data:application/octet-stream;base64," +
                   base64.b64encode(blob).decode("ascii"),
        }],
        "bufferViews": views,
        "accessors": accessors,
    }
    return doc


def write_gltf(model: Model3D, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf8", newline="\n") as handle:
        json.dump(to_gltf(model), handle, indent=1, sort_keys=True)
    return target


def to_viewer_payload(model: Model3D, street_bearing: float | None = None) -> dict:
    """Compact shape for the browser viewer.

    Short keys and pre-decomposed wall panels, so the page does no geometry
    work of its own beyond building boxes. Window-view data rides along, since
    it is what distinguishes one room from another.
    """
    from ingest.mp3d.views import (
        street_bearing_from_entrances, view_summary, views_payload,
    )

    if street_bearing is None:
        street_bearing = street_bearing_from_entrances(model)
    return {
        "views": views_payload(model, street_bearing),
        "viewStats": view_summary(model, street_bearing),
        "schema": model.schema,
        "id": model.model_id,
        "name": model.name,
        "yaw": model.yaw_deg,
        "tier": model.tier,
        "bounds": model.bounds,
        "stats": model.stats,
        "notes": list(model.notes),
        "slabs": [
            {"f": s.floor, "label": s.label, "z0": s.z0, "h": s.height_m,
             "src": s.elevation_source}
            for s in model.slabs
        ],
        "rooms": [
            {"id": r.id, "f": r.floor, "cls": r.classification,
             "p": [list(p) for p in r.polygon], "c": list(r.centroid),
             "a": r.area_m2}
            for r in model.rooms
        ],
        "walls": [
            {"id": w.id, "f": w.floor, "a": list(w.a), "b": list(w.b),
             "th": w.thickness_m, "ext": w.exterior, "iv": w.open_boundary,
             "op": [{"t": o.kind[0], "t0": o.t0, "t1": o.t1, "z0": o.z0, "z1": o.z1}
                    for o in w.openings]}
            for w in model.walls
        ],
    }


def write_viewer_payload(
    model: Model3D, path: str | Path, var: str = "MP3D",
    street_bearing: float | None = None,
) -> Path:
    """Write the viewer payload as a JS file assigning ``window.<var>``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(
        to_viewer_payload(model, street_bearing), separators=(",", ":"), sort_keys=True)
    with open(target, "w", encoding="utf8", newline="\n") as handle:
        handle.write(f"window.{var}={body};\n")
    return target
