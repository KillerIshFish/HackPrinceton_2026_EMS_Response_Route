# Window views and the site map: how to bring them into another branch

This addition does three things:

1. **Local map data.** OpenStreetMap roads, buildings, car parks and named places around every
   property are read from disk in about 2 ms, instead of from the public Overpass server
   (which was taking 25 to 60 s per request, or failing).
2. **Window views.** Every windowed room of a single-building property gets a
   `Room.window_facing` of `street`, `parking_lot`, `residential` or `adjacent_building`,
   so view clues ("I can see the car park") can narrow the search.
3. **A street map in the web app.** The "Site and approach" panel has an Aerial / Map switch.
   The map is drawn from the same features the view logic uses.

It was built on `integration/unified-model` and depends on `ingest/unified/` and
`ingest/mp3d/` being present.

## The map data is in this repo. Do not download it again.

The file `data/osm/site_pack.sqlite.gz` (36 MB) is committed. It holds everything within
230 m of each of the 3,339 properties in the kit: 383,490 map elements.

- **You do not need to run anything.** The first lookup unpacks it to
  `data/processed/osm_index.sqlite` (198 MB, gitignored) if no index is there.
- To unpack it explicitly: `uv run python -m ingest.unified.osm_local --unpack`
- To bring only the map data into another branch:

  ```bash
  git checkout integration/unified-model -- data/osm/site_pack.sqlite.gz .gitignore
  ```

  The `.gitignore` change matters: `data/**` is ignored, and one added line
  (`!data/osm/site_pack.sqlite.gz`) lets this file through.

**Do not** run `--download` or the full index build unless you need addresses that are not
in the kit. That path fetches 1.4 GB, takes about 16 minutes and writes a 3 GB index. The
pack was checked against the full index on 300 properties and returned identical elements
for all of them.

What the pack does not cover: an address more than 60 m from any kit property. Those
lookups fall back to the public Overpass server, as before.

## Files to bring across

New files (copy as they are):

| File | What it is |
|---|---|
| `data/osm/site_pack.sqlite.gz` | The map data |
| `ingest/unified/osm_local.py` | Reads the index; builds, packs and unpacks it |
| `ingest/unified/surroundings.py` | Map elements to features; rays from windows to a view class |
| `tests/test_unified_surroundings.py` | Tests, no network |
| `web/src/components/SiteMapLayer.tsx` | The street map layer |
| `docs/osm-site-map.md` | This file |

Changed files (merge, do not overwrite, if the other branch has touched them):

| File | What changed |
|---|---|
| `.gitignore` | One exception for the site pack |
| `ingest/unified/context.py` | `fetch_site` tries the local index first; the Overpass query also asks for car parks and residential land; the sidecar gains `surroundings`; `CONTEXT_VERSION` is 5 |
| `ingest/unified/project.py` | `to_building` sets `window_facing` from rays, falling back to the street bearing |
| `ingest/unified/pipeline.py` | `_meta` passes `surroundings` to the web app under `site` |
| `web/src/types.ts` | `SiteFeature`, `SiteSurroundings`, `site.surroundings` |
| `web/src/components/AerialSite.tsx` | `mode` prop, map layer, larger view, green staging marker |
| `web/src/components/AerialApproach.tsx` | Aerial / Map switch, new layout, title "Site and approach" |
| `web/src/App.tsx` | The panel moved above the 3D model |
| `pyproject.toml`, `uv.lock` | `osmium` added to the dev group |
| `README.md` | Two short sections |

To take everything in one step from a branch that has not diverged in these files:

```bash
git checkout integration/unified-model -- \
  data/osm/site_pack.sqlite.gz .gitignore \
  ingest/unified/osm_local.py ingest/unified/surroundings.py \
  ingest/unified/context.py ingest/unified/project.py ingest/unified/pipeline.py \
  tests/test_unified_surroundings.py \
  web/src/components/SiteMapLayer.tsx web/src/components/AerialSite.tsx \
  web/src/components/AerialApproach.tsx web/src/App.tsx web/src/types.ts \
  docs/osm-site-map.md
```

If the other branch has its own edits to `context.py`, `project.py`, `App.tsx` or the two
Aerial components, merge the branch instead and resolve by hand; the notes below say what
each change must still do afterwards.

## Dependencies

- **To use the pack: none.** `osm_local.py` needs only the standard library for reading.
- **To rebuild the full index: `osmium`** (pyosmium 4). It is in the dev group, so
  `uv sync` installs it. One test is skipped when it is missing.
- The web app needs no new packages.

## How the pieces connect

```
data/osm/site_pack.sqlite.gz
   -> osm_local.fetch_site(lat, lon)            Overpass-shaped {"elements": [...]}
   -> context.fetch_site                        local first, Overpass otherwise
   -> surroundings.surroundings_from_payload    features (class + line or area) and places
   -> surroundings.mark_own                     flags the map's tracing of the modelled building
   -> context sidecar  data/processed/context/<id>.json   ["surroundings"]
        -> project.to_building                  rays from each window -> Room.window_facing
        -> pipeline._meta -> /buildings/{id}/meta -> site.surroundings -> SiteMapLayer
```

Points that are easy to break when merging:

- `osm_local.fetch_site` returns the same shape as an Overpass `out geom tags` answer, so
  `street_from_payload` and `building_from_payload` in `context.py` read it unchanged. Keep
  it that way and nothing downstream needs to know which source answered.
- The map data lives in the sidecar, never in `BuildingModel`, which must stay byte-identical
  for the same inputs.
- `mark_own` must run before rays are cast. Without it a window "sees" the map's outline of
  its own building and every room comes out as `adjacent_building`.
- Views are skipped when `layout_placed` is false (a property with several buildings),
  because those positions on the ground are invented.
- A view class is the first thing most rays meet within 60 m: five rays per window, spread
  over 60 degrees. When no ray meets anything, the old rule applies: a window within 60
  degrees of the street bearing faces the street.
- The street map and the aerial image share one coordinate system in `AerialSite.tsx`
  (tile-mosaic pixels), so the outline, entrances and staging marker need no second
  projection. `SiteMapLayer` only replaces the image tiles.

## After merging

1. `uv sync`
2. `uv run pytest tests/test_unified_surroundings.py tests/test_unified.py` (19 tests)
3. `uv run python -m ingest.unified.osm_local 34.0322954 -118.4522354`
   should print about 130 elements in a few milliseconds.
4. Regenerate any building made before the merge, for example
   `uv run python -m ingest.unified 246948`. Older sidecars are version 4 and are rebuilt
   automatically, but the building and meta files are only rewritten when the pipeline runs.
5. In `web/`: `npm ci`, then `npm run typecheck`.
6. Start both servers, open `http://localhost:5173/?building=prop-246948`, and switch the
   "Site and approach" panel to Map. Streets should be named and the amber outline should
   sit in the same place as on the aerial view.

## Rebuilding the data (only if the kit's properties change)

```bash
uv run python -m ingest.unified.osm_local --download   # 1.4 GB into data/raw/osm/
uv run python -m ingest.unified.osm_local              # full index, about 16 min, 3 GB
uv run python -m ingest.unified.osm_local --pack       # rewrites data/osm/site_pack.sqlite.gz
```

`--pack` reads property coordinates from `data/processed/property_index.sqlite`, so the
address index must exist first. Commit the new pack. A machine that has the full index
keeps using it; `--unpack --force` replaces it with the pack.

## Known limits

- **Accuracy is unchecked.** On a 40-property sample, 94% of windowed rooms in placed
  buildings got a class, but the classes have not been compared with imagery.
- **Car parks are often unmapped.** 11 of 28 placed buildings had one nearby. Bundy Olympic's
  parking deck is not in OpenStreetMap, so its south side reads as `street`.
- **Some buildings come out as `adjacent_building` for every room.** In the cases examined a
  neighbouring outline overlapped the fitted model box. Check these on the Map view.
- **`Room.window_facing` holds one class.** A corner room with glass on two sides gets the
  class most of its rays agree on.
- **Named places are drawn but not scored.** Using them for landmark clues is the next step.

## Licence

The pack is OpenStreetMap data, (c) OpenStreetMap contributors, under the Open Database
Licence (ODbL). The map view carries the attribution. Redistribution in the repo is allowed
with that attribution; keep the credit if the view is restyled.
