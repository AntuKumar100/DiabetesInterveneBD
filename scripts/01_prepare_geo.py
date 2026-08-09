"""
Prepare the district geometry layer.

Why this script exists
----------------------
The BDHS district labels and the OSM/HDX administrative boundaries do not agree
on spelling. Bangladesh officially re-romanised several district names in 2018
(Chittagong -> Chattogram, Barisal -> Barishal, Comilla -> Cumilla, Jessore ->
Jashore, Bogra -> Bogura). The boundary file still carries the pre-2018 forms,
plus one outright typo ("Brahamanbaria") and one short form ("Nawabganj" for
Chapai Nawabganj). Seven of sixty-four districts fail an exact join.

Silently dropping those seven would delete ~11% of the map, including
Chattogram — the second largest district in the survey. So the alias table is
explicit, version-controlled, and asserted at build time: if the join ever
degrades, the build fails loudly instead of rendering a map with holes in it.

The raw boundary file is ~4.6 MB, which is slow to ship to a browser on every
rerun. We simplify with a 0.004 degree tolerance (roughly 400 m at this
latitude). At a national zoom level that is well below one screen pixel, and it
cuts the payload by about an order of magnitude.

Run:
    python scripts/01_prepare_geo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from shapely.geometry import mapping, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import GEOJSON_OUT, CENTROIDS_OUT, RAW_GEOJSON, DISTRICT_ALIASES  # noqa: E402

SIMPLIFY_TOLERANCE_DEG = 0.004


def normalise(name: str) -> str:
    """Boundary-file name -> BDHS name. Lowercase, then apply the alias table."""
    key = name.strip().lower()
    return DISTRICT_ALIASES.get(key, key)


def main() -> int:
    if not RAW_GEOJSON.exists():
        print(f"Missing {RAW_GEOJSON}.", file=sys.stderr)
        print("Download it once with:", file=sys.stderr)
        print(
            "  curl -L -o data/raw/bd_districts_raw.geojson \\\n"
            "    https://raw.githubusercontent.com/nuhil/bangladesh-geocode/"
            "master/geojson/districts.geojson",
            file=sys.stderr,
        )
        return 1

    raw = json.loads(RAW_GEOJSON.read_text())
    features, centroids = [], {}

    for feat in raw["features"]:
        src_name = feat["properties"]["ADM2_EN"]
        name = normalise(src_name)

        geom = shape(feat["geometry"])
        if not geom.is_valid:
            # buffer(0) is the standard trick for self-intersecting admin polygons
            geom = geom.buffer(0)
        simple = geom.simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
        if simple.is_empty:
            simple = geom  # never let simplification delete a district

        pt = simple.representative_point()  # inside the polygon even when concave
        centroids[name] = {"lat": round(pt.y, 5), "lon": round(pt.x, 5)}

        features.append(
            {
                "type": "Feature",
                "id": name,
                "properties": {
                    "district": name,
                    "source_name": src_name,
                    "renamed": name != src_name.strip().lower(),
                },
                "geometry": mapping(simple),
            }
        )

    if len(features) != 64:
        print(f"Expected 64 districts, got {len(features)}.", file=sys.stderr)
        return 1

    out = {"type": "FeatureCollection", "features": features}
    GEOJSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    GEOJSON_OUT.write_text(json.dumps(out, separators=(",", ":")))
    CENTROIDS_OUT.write_text(json.dumps(centroids, indent=2, sort_keys=True))

    # National outline, used as a hairline under the choropleth so that
    # low-opacity (high-uncertainty) districts still read as part of a country.
    outline = unary_union([shape(f["geometry"]) for f in features])
    (GEOJSON_OUT.parent / "bd_outline.geojson").write_text(
        json.dumps(mapping(outline.simplify(0.01, preserve_topology=True)),
                   separators=(",", ":"))
    )

    before = RAW_GEOJSON.stat().st_size / 1e6
    after = GEOJSON_OUT.stat().st_size / 1e6
    renamed = sum(f["properties"]["renamed"] for f in features)
    print(f"64 districts written  ({before:.1f} MB -> {after:.1f} MB)")
    print(f"{renamed} names remapped via the alias table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
