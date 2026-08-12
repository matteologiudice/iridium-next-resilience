"""Synthetic user terminals, distributed across three regions.

Generates the twenty-four compromised terminals used by the bandwidth exhaustion scenario, eight in
each of three regions, by sampling latitude and longitude uniformly inside a bounding box from a
generator with a fixed seed.

Why three regions rather than one
---------------------------------
A group of terminals confined to a single region cannot contend with both legitimate flows. The
satellites a high-latitude station sees at any given moment serve widely separated bands of
longitude, because a near-polar constellation sweeps the whole globe, so terminals in one region
reach the satellites overhead that region and no others.

A scan over a latitude and longitude grid across the northern hemisphere makes the effect concrete.
Within a representative visibility window, the legitimate traffic into Svalbard uses three
satellites, and each is naturally reachable only from a specific band:

  Europe and western Russia          longitude   -2 to   46
  North America, Pacific coast       longitude -146 to  -98
  North America, central and east    longitude  -86 to  -38

The alignment is not accidental. Fucino, at longitude 13.6, falls in the first band, and Tempe, at
longitude -111.9, falls in the second. A botnet able to contend with both legitimate flows must
therefore be spread across continents rather than concentrated in one place.

Sampling a rectangle in this way places some terminals at sea, which matches how the constellation
is used: the simulator resolves visibility from position alone, with no notion of land or water.
"""

import numpy as np

_SEED = 42
_N_PER_REGION = 8

_REGIONS = {
    "Europa/Russia occidentale (-> satellite 52)": {"lat": (45.0, 65.0), "lon": (0.0, 45.0)},
    "USA - costa Pacifico (-> satellite 75)": {"lat": (30.0, 55.0), "lon": (-125.0, -100.0)},
    "USA - centro/est (-> satellite 97)": {"lat": (30.0, 55.0), "lon": (-85.0, -60.0)},
}

_rng = np.random.default_rng(_SEED)

_rows = []
_names = []
for region_name, bounds in _REGIONS.items():
    lats = _rng.uniform(bounds["lat"][0], bounds["lat"][1], size=_N_PER_REGION)
    lons = _rng.uniform(bounds["lon"][0], bounds["lon"][1], size=_N_PER_REGION)
    for lat, lon in zip(lats, lons):
        _rows.append((lat, lon, 0.0))
        _names.append(f"Synthetic user terminal [{region_name}] (lat={lat:.2f}, lon={lon:.2f})")

UT_SYNTHETIC_MULTIREGION = np.array(_rows)
UT_SYNTHETIC_MULTIREGION_NAMES = _names

assert len(UT_SYNTHETIC_MULTIREGION) == _N_PER_REGION * len(_REGIONS) == len(UT_SYNTHETIC_MULTIREGION_NAMES)

if __name__ == "__main__":
    print(f"Synthetic user terminals: {len(UT_SYNTHETIC_MULTIREGION)} positions generated (seed={_SEED})")
    for region_name, bounds in _REGIONS.items():
        print(f"  {region_name}: lat {bounds['lat']}, lon {bounds['lon']}, {_N_PER_REGION} punti")
