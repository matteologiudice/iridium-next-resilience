"""Composes the constellation with both operator ground stations and user terminals.

Builds IridiumUserTerminalMultiConstellation: the same 66 satellites and eight reconstructed ground
stations used everywhere else, plus the twenty-four synthetic user terminals as a second, distinct
group of ground-side nodes.

Why two separate ground constellations
--------------------------------------
MultiConstellation.add_constellation() and add_ill_helper() are generic and may be called more than
once, so there is no need to merge the stations and the terminals into a single array. Keeping them
apart groups node IDs by role: the eight real stations take the first IDs, the terminals the next
twenty-four, and the satellites the rest. The attack and analysis scripts can then tell a legitimate
ground station from a compromised terminal without carrying an extra mapping around.

The terminal group is given an explicit name. The default would collide with the first group on the
same IDHelper, since GroundConstellation names its nodes "{name}_{index}" and two groups sharing a
name would try to assign the same identifiers twice.
"""

import os
import sys
from typing import Any

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "model_and_baseline"),
    os.path.dirname(os.path.abspath(__file__)),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dsns.multiconstellation import MultiConstellation
from dsns.presets import ground_constellation, iridium_constellation, ground_ill_helper
from dsns.helpers import IDHelper

from ground_stations_reconstructed import IRIDIUM_GROUND_STATIONS_RECONSTRUCTED  # noqa: E402
from ut_synthetic_multiregion import UT_SYNTHETIC_MULTIREGION  # noqa: E402


class IridiumUserTerminalMultiConstellation(MultiConstellation):
    """The reconstructed constellation plus a second group of twenty-four user terminals.

    The terminals are added as their own ground-side group, with an independent ILL helper toward
    the same satellite constellation. Both groups use the same minimum elevation of 8.2 degrees: a
    user terminal has no more sensitive an antenna than an operator ground station, so there is no
    reason to give it a different threshold.
    """

    def __init__(self, iridium_kwargs: dict[str, Any] = {}):
        super().__init__()

        self.id_helper = IDHelper()

        # Group 1: the eight reconstructed ground stations (IDs 0 to 7)
        self.ground_constellation = ground_constellation(
            self.id_helper,
            ground_station_positions=IRIDIUM_GROUND_STATIONS_RECONSTRUCTED,
        )

        # Group 2: the twenty-four synthetic user terminals (IDs 8 to 31). The name is set
        # explicitly: the default would collide with group 1 on the same IDHelper, since nodes are
        # named "{name}_{index}" and two groups sharing a name would assign "ground_0" twice,
        # raising a ValueError.
        self.ut_constellation = ground_constellation(
            self.id_helper,
            ground_station_positions=UT_SYNTHETIC_MULTIREGION,
            name="ut",
        )

        # Satellites (IDs 32 to 97)
        self.iridium_constellation = iridium_constellation(self.id_helper, **iridium_kwargs)

        # ILL helper indipendenti per ciascun gruppo di nodi a terra
        self.ground_ill_helper = ground_ill_helper(
            self.ground_constellation, self.iridium_constellation, min_elevation=8.2
        )
        self.ut_ill_helper = ground_ill_helper(
            self.ut_constellation, self.iridium_constellation, min_elevation=8.2
        )

        self.add_constellation(self.ground_constellation)
        self.add_constellation(self.ut_constellation)
        self.add_constellation(self.iridium_constellation)
        self.add_ill_helper(self.ground_ill_helper)
        self.add_ill_helper(self.ut_ill_helper)


if __name__ == "__main__":
    m = IridiumUserTerminalMultiConstellation()
    m.update(0.0)
    n_gs = len(m.ground_constellation.satellites.ids)
    n_ut = len(m.ut_constellation.satellites.ids)
    n_sat = len(m.iridium_constellation.satellites.ids)
    print(f"Iridium + UT scenario: {len(m.satellites.ids)} nodi totali "
          f"({n_gs} ground stations + {n_ut} user terminals + {n_sat} satellites), "
          f"{len(m.isls)} ISL, {len(m.ills)} ILL")
    print(f"Ground station ID range: 0-{n_gs - 1}")
    print(f"User terminal ID range: {n_gs}-{n_gs + n_ut - 1}")
    print(f"Satellite ID range: {n_gs + n_ut}-{n_gs + n_ut + n_sat - 1}")
