# District name reconciliation

The BDHS extract and the administrative boundary file disagree on seven of
sixty-four district names. Left unhandled, seven districts fall out of the join
and the map renders with holes in it — including Chattogram, the second-largest
district in the survey.

The table lives in `src/config.py` as `DISTRICT_ALIASES` and is applied in
`scripts/01_prepare_geo.py`. The build asserts a count of 64 and fails loudly
if the join ever degrades.

| Boundary file (`ADM2_EN`) | BDHS extract | Why |
|---|---|---|
| Barisal | barishal | 2018 romanisation reform |
| Bogra | bogura | 2018 romanisation reform |
| Chittagong | chattogram | 2018 romanisation reform |
| Comilla | cumilla | 2018 romanisation reform |
| Jessore | jashore | 2018 romanisation reform |
| Brahamanbaria | brahmanbaria | Typo in the source boundary file — the correct romanisation has no second *a* after *Brah* |
| Nawabganj | chapai nawabganj | Boundary file uses the short form; the district's full name distinguishes it from Nawabganj upazila in Dhaka division |

## Background

In 2018 the Bangladesh government standardised the English romanisation of
several place names to bring them closer to Bangla pronunciation. The BDHS 2022
extract uses the post-reform spellings. Most publicly available boundary files
still carry the pre-reform ones, because they descend from older OSM and HDX
exports.

This is a routine hazard when joining Bangladeshi administrative data across
sources, and it is worth checking every time rather than assuming a clean join.

## Verifying the join

```bash
python scripts/01_prepare_geo.py
# expected output:
#   64 districts written  (4.6 MB -> 0.5 MB)
#   7 names remapped via the alias table
```

To inspect which names failed before aliasing:

```python
import json, pandas as pd
geo = {f["properties"]["ADM2_EN"].lower()
       for f in json.load(open("data/raw/bd_districts_raw.geojson"))["features"]}
data = set(pd.read_excel("data/raw/bdhs_diabetes.xls", engine="xlrd")["District"])
print(sorted(data - geo))   # the seven above
```

## Division names

Division names differ too — the boundary file uses `Chittagong` where the
extract uses `chattogram`, and similarly for Barisal. Divisions are read from
the survey extract rather than from the boundary file, so no alias table is
needed there. If you ever switch to boundary-file divisions, this is the first
thing that will break.
