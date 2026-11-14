"""CLI for the Matterport geometry -> 3D pipeline.

    uv run python -m ingest.mp3d <model_id>                     # build, print summary
    uv run python -m ingest.mp3d <model_id> -o out.json         # canonical JSON
    uv run python -m ingest.mp3d <model_id> --gltf out.gltf     # standard 3D file
    uv run python -m ingest.mp3d <model_id> --viewer data3d.js  # browser payload
    uv run python -m ingest.mp3d <model_id> --preflight         # just classify
    uv run python -m ingest.mp3d <model_id> --layers            # per-layer report
    uv run python -m ingest.mp3d <model_id> --views             # window-view analysis
    uv run python -m ingest.mp3d <model_id> --scan -o scan.json # scan-cluster fallback
    uv run python -m ingest.mp3d <model_id> --aerial            # aerial sidecar
    uv run python -m ingest.mp3d --from-payload saved.json      # offline rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ingest.mp3d.client import MatterportClient, MatterportError
from ingest.mp3d.clusters import build_scan_model
from ingest.mp3d.export import write_gltf, write_viewer_payload
from ingest.mp3d.fetch import fetch_geometry, load_payload, save_payload
from ingest.mp3d.layers import layer_report
from ingest.mp3d.model3d import build_model, model_from_payload
from ingest.mp3d.preflight import preflight
from ingest.mp3d.views import (
    classify_facing, distinctiveness, room_views,
    street_bearing_from_entrances, view_summary,
)


def _summary(model) -> str:
    lines = [
        f"{model.name or '(unnamed)'}  [{model.model_id}]  tier {model.tier}",
        f"  units {model.units} | yaw {model.yaw_deg:.2f} deg"
        f" | bounds x{model.bounds['x']} y{model.bounds['y']} z{model.bounds['z']}",
        "  slabs:",
    ]
    for slab in model.slabs:
        lines.append(
            f"    floor {slab.floor} ({slab.label}): z0={slab.z0:.3f} m"
            f" h={slab.height_m:.3f} m  [{slab.elevation_source}"
            f", n={slab.elevation_samples}]"
        )
    stats = model.stats
    lines.append(
        f"  rooms {stats['room_plates']}/{stats['rooms_in_payload']} kept"
        f" (dropped: {stats['dropped_no_ring']} no-ring,"
        f" {stats['dropped_void']} void, {stats['dropped_below_min_area']} tiny)"
    )
    lines.append(
        f"  walls {stats['walls']} ({stats['walls_exterior']} exterior,"
        f" {stats['walls_open_boundary']} open boundary,"
        f" {stats['walls_thickness_measured']} measured thickness)"
    )
    lines.append(
        f"  openings {stats['openings']} {stats['openings_by_kind']}"
        f" | floor area {stats['floor_area_m2']} m2"
    )
    for note in model.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


def _scan_summary(scan) -> str:
    lines = [
        f"{scan.name or '(unnamed)'}  [{scan.model_id}]  {scan.schema}",
        f"  SCAN-CLUSTER build - positions and connectivity, no wall geometry",
    ]
    for slab in scan.slabs:
        lines.append(
            f"    floor {slab.floor} ({slab.label}): raw z={slab.raw_z:.3f}"
            f"  [{slab.source}, n={slab.sample_size}]"
            f"  height={scan.floor_heights.get(slab.floor)} m"
        )
    s = scan.stats
    lines.append(
        f"  zones {s['rooms']} ({s['rooms_with_classification']} classified,"
        f" {s['rooms_with_area']} with area) | anchors {s['anchors']}"
    )
    lines.append(
        f"  links {s['room_links']} from {s['anchor_edges']} anchor edges"
        f" ({s['cross_floor_links']} cross-floor)"
    )
    for note in scan.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


def _views_summary(model, street_bearing) -> str:
    summary = view_summary(model, street_bearing)
    views = room_views(model)
    shared = distinctiveness(views)
    reference = (
        street_bearing
        if street_bearing is not None
        else street_bearing_from_entrances(model)
    )
    lines = ["", "WINDOW VIEWS"]
    lines.append(
        f"  {summary['rooms_with_windows']}/{summary['rooms']} rooms have windows"
        f" | {summary['windows']} windows, {summary['glazed_area_m2']} m2 glazed"
        f" | {summary['corner_rooms']} corner rooms"
    )
    lines.append(
        f"  street bearing: {reference if reference is not None else 'unknown'}"
        f" | facing {summary['facing']}"
    )
    lines.append(f"  sectors faced: {summary['sectors']}")
    lines.append(
        f"  uniquely identified by view signature:"
        f" {summary['uniquely_identified_rooms']}/{summary['rooms']}"
        f" ({100 * summary['uniquely_identified_share']:.0f}%)"
    )
    lines.append(f"  window sizes {summary['size_classes']}"
                 f" | sills {summary['sill_classes']}")
    lines.append("  per room:")
    for view in views:
        if not view.has_windows:
            continue
        facing = classify_facing(view, reference)
        lines.append(
            f"    [x{shared[view.room_id]}] {facing:11s} {view.describe()}"
        )
    blind = [v for v in views if not v.has_windows]
    if blind:
        lines.append(
            f"  {len(blind)} windowless room(s): "
            + ", ".join(sorted({v.classification for v in blind}))
        )
    return "\n".join(lines)


def _aerial_summary(model, street_bearing) -> str:
    """Build the sidecar and report it. Network: OSM plus Vexcel."""
    from ingest.mp3d.aerial import AerialError, build_aerial_ref, save_sidecar
    from ingest.mp3d.views import street_bearing_from_entrances

    geo = getattr(model, "geolocation", None)
    if not geo:
        return ("\nAERIAL\n  this model carries no geolocation, so there is nothing "
                "to look up - the sidecar is optional per model (about 79% of the "
                "corpus has coordinates)")
    entrance = (street_bearing if street_bearing is not None
                else street_bearing_from_entrances(model))
    try:
        ref, verdict = build_aerial_ref(
            model.model_id, geo[0], geo[1], model.yaw_deg,
            entrance_bearing_deg=entrance)
    except AerialError as exc:
        return f"\nAERIAL\n  unavailable: {exc}"
    save_sidecar(ref)
    lines = ["", "AERIAL SIDECAR (not part of the model's canonical JSON)"]
    lines.append(f"  street bearing : {ref.street_bearing_deg} deg via "
                 f"{ref.bearing_source} [{ref.bearing_confidence}"
                 f"{' AMBIGUOUS' if ref.bearing_ambiguous else ''}]")
    lines.append(f"  {ref.bearing_note}")
    lines.append(f"  entrance bearing: {entrance} deg (frontage)")
    lines.append(f"  gap            : {verdict.get('gap_deg')} deg")
    lines.append(f"  verdict        : {verdict.get('verdict')}")
    lines.append(f"  recommended    : {verdict.get('recommended_bearing_deg')} deg "
                 f"via {verdict.get('recommended_source')}")
    if ref.local_path:
        lines.append(f"  imagery        : {ref.local_path} "
                     f"({ref.collection}, captured {ref.captured_at})")
        lines.append(f"  sha256         : {ref.content_sha256}")
    if ref.oblique_directions:
        lines.append(f"  obliques       : {', '.join(ref.oblique_directions)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ingest.mp3d",
        description="Build a deterministic 3D model from a Matterport model id.",
    )
    parser.add_argument("model_id", nargs="?", help="Matterport model id")
    parser.add_argument("--preflight", action="store_true",
                        help="only report whether the model can be built")
    parser.add_argument("-o", "--out", help="write canonical JSON here")
    parser.add_argument("--gltf", help="also write a glTF 2.0 file here")
    parser.add_argument("--viewer", help="also write the browser payload (.js) here")
    parser.add_argument("--save-payload", help="save the raw API payload here")
    parser.add_argument("--from-payload",
                        help="build from a saved payload instead of calling the API")
    parser.add_argument("--layers", action="store_true",
                        help="report which data layers the model carries")
    parser.add_argument("--views", action="store_true",
                        help="report window views and how well they tell rooms apart")
    parser.add_argument("--scan", action="store_true",
                        help="build the scan-cluster model (for models with no wall geometry)")
    parser.add_argument("--street-bearing", type=float, default=None,
                        help="reference bearing for the street, degrees east of north")
    parser.add_argument("--aerial", action="store_true",
                        help="build the aerial sidecar (street bearing, parcel, ortho)")
    parser.add_argument("--quiet", action="store_true", help="suppress the summary")
    args = parser.parse_args(argv)

    if not args.model_id and not args.from_payload:
        parser.error("give a model_id, or --from-payload")

    try:
        if args.from_payload:
            payload = load_payload(args.from_payload)
            model = model_from_payload(payload, args.model_id)
        else:
            if args.preflight:
                print(preflight(args.model_id).describe())
                return 0
            if args.layers:
                print(layer_report(args.model_id).describe())
                return 0
            client = MatterportClient()
            if args.scan:
                scan = build_scan_model(args.model_id, client=client)
                if not args.quiet:
                    print(_scan_summary(scan))
                if args.out:
                    print(f"json   -> {scan.to_json_file(args.out)}")
                elif not args.quiet:
                    print("  (pass -o to write the scan model)")
                return 0
            if args.save_payload:
                payload = fetch_geometry(args.model_id, client=client)
                save_payload(payload, args.save_payload)
                print(f"payload -> {args.save_payload}")
                model = model_from_payload(payload, args.model_id)
            else:
                model = build_model(args.model_id, client=client)
    except MatterportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(_summary(model))
    if args.views:
        print(_views_summary(model, args.street_bearing))
    if args.aerial:
        print(_aerial_summary(model, args.street_bearing))

    if args.out:
        print(f"json   -> {model.to_json_file(args.out)}")
    if args.gltf:
        print(f"gltf   -> {write_gltf(model, args.gltf)}")
    if args.viewer:
        print(f"viewer -> {write_viewer_payload(model, args.viewer)}")
    if not any((args.out, args.gltf, args.viewer)):
        print("  (no output path given; pass -o/--gltf/--viewer to write files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
