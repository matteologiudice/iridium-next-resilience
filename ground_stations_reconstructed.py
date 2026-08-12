"""Reconstructed coordinates for the Iridium NEXT ground segment.

This is a reconstruction, not an official dataset. Neither Iridium Communications nor its
predecessor has ever published a complete list of coordinates for the constellation's ground
stations. What is publicly available falls into two kinds: place names, given in corporate filings
and press releases without coordinates, and precise positions for individual facilities, buried in
national regulatory filings. The eight positions below were assembled by cross-referencing SEC
filings, Iridium press releases, FCC engineering statements and, where no regulatory filing was
available, independent reporting on the specific facility.

Coordinate confidence is not uniform, and the difference is recorded per site rather than averaged
away:

  Tempe, Chandler, Fairbanks   Exact. Read from the Latitude and Longitude fields of FCC filings
                               SES-MOD-20170208-00135/00136/00138, cross-checked against the
                               original licence filings from 1996, 2003 and 2005, which agree to
                               within fractions of an arcsecond.
  Svalbard                     Exact. Published coordinates of the SvalSat/Plataberget station,
                               where Iridium is explicitly named as a customer.
  Fucino                       Exact. Published coordinates of the Fucino Space Centre.
  Wahiawa                      Exact. Published coordinates of the NCTAMS PAC Wahiawa Annex, which
                               hosts the United States government gateway operated for the
                               Department of Defense.
  Punta Arenas                 Approximate. The operating partner states that four of its six
                               antennas at the site carry Iridium traffic, so the area is
                               confirmed, but the coordinates are those of a neighbouring facility
                               in the same industrial zone north of the city.
  Izhevsk                      Approximate, at city level. The site is well documented, having
                               opened in November 2016 with three Ka-band antennas, but no street
                               address or filing giving a precise position was found.

The Leesburg, Virginia network operations centre is excluded. It coordinates the constellation but
is not itself an antenna facility connecting to the satellites.

Format follows dsns.helpers: an array of [latitude, longitude, height in metres], so it can be
passed directly as ground_station_positions to dsns.presets.ground_constellation(). No DSNS source
file is modified; only a different array is passed to a function that already accepts one.
"""

import numpy as np

IRIDIUM_GROUND_STATIONS_RECONSTRUCTED = np.array([
    (33.342222, -111.896111, 362.7),   # Tempe, Arizona, USA. Commercial gateway (exact, FCC E960131)
    (33.266444, -111.881639, 366.0),   # Chandler, Arizona, USA. TT&C (exact, FCC E960244)
    (64.817556, -147.724111, 128.0),   # Fairbanks, Alaska, USA. TT&C (exact, FCC E050282)
    (78.229772, 15.407786, 0.0),       # Svalbard (SvalSat/Plataberget), Norway. Gateway (exact)
    (-52.930000, -70.850000, 0.0),     # Punta Arenas, Chile. Gateway (area confirmed, approximate)
    (41.978889, 13.600833, 0.0),       # Fucino, Italy. European gateway, Telespazio (exact)
    (56.852600, 53.211400, 0.0),       # Izhevsk, Udmurt Republic, Russia. Gateway (city level)
    (21.520833, -157.995278, 0.0),     # Wahiawa, Hawaii, USA. Government gateway, DoD/DISA (exact)
])

IRIDIUM_GROUND_STATIONS_NAMES = [
    "Tempe, Arizona, USA (commercial gateway, exact coordinates from FCC filing E960131)",
    "Chandler, Arizona, USA (TT&C, exact coordinates from FCC filing E960244)",
    "Fairbanks, Alaska, USA (TT&C, exact coordinates from FCC filing E050282)",
    "Svalbard (SvalSat/Plataberget), Norway (gateway, exact coordinates, Iridium a named customer)",
    "Punta Arenas, Chile (gateway, area confirmed, coordinates approximate)",
    "Fucino, Italy (European gateway, Telespazio Space Centre, exact coordinates)",
    "Izhevsk, Udmurt Republic, Russia (gateway operating since 2016, city-level coordinates)",
    "Wahiawa, Hawaii, USA (government gateway, DoD/DISA, NCTAMS PAC Wahiawa Annex, exact coordinates)",
]

assert len(IRIDIUM_GROUND_STATIONS_RECONSTRUCTED) == len(IRIDIUM_GROUND_STATIONS_NAMES)

if __name__ == "__main__":
    print(f"Iridium NEXT: {len(IRIDIUM_GROUND_STATIONS_RECONSTRUCTED)} reconstructed ground stations")
    for name in IRIDIUM_GROUND_STATIONS_NAMES:
        print(f"  {name}")
