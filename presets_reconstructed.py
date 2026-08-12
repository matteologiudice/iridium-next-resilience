"""Composes the Iridium NEXT constellation with the reconstructed ground segment.

DSNS provides dsns.presets.IridiumMultiConstellation, which builds its ground constellation
internally by calling ground_constellation(id_helper) with no way to supply ground stations of its
own: the result is the simulator's synthetic default of 256 procedurally generated positions with
no correspondence to real sites.

The class below reproduces that same composition, using the same add_constellation and
add_ill_helper pattern and the same minimum elevation angle, but calls
ground_constellation(id_helper, ground_station_positions=...) directly. That function is already
public in dsns.presets and already forwards its keyword arguments to GroundConstellation, so the
default is overridden without touching any DSNS source file: this module only composes existing
public classes and functions with different arguments.

The minimum elevation of 8.2 degrees is Iridium's own disclosed value and is left unchanged.
"""

from typing import Any

from dsns.multiconstellation import MultiConstellation
from dsns.presets import ground_constellation, iridium_constellation, ground_ill_helper
from dsns.helpers import IDHelper

from ground_stations_reconstructed import IRIDIUM_GROUND_STATIONS_RECONSTRUCTED


class IridiumReconstructedMultiConstellation(MultiConstellation):
    """Iridium NEXT with the eight reconstructed ground stations.

    Equivalent to dsns.presets.IridiumMultiConstellation, except that the ground segment is built
    from the reconstructed positions rather than from the simulator's 256 synthetic defaults.
    """

    def __init__(self, iridium_kwargs: dict[str, Any] = {}):
        super().__init__()

        self.id_helper = IDHelper()

        self.ground_constellation = ground_constellation(
            self.id_helper,
            ground_station_positions=IRIDIUM_GROUND_STATIONS_RECONSTRUCTED,
        )
        self.iridium_constellation = iridium_constellation(self.id_helper, **iridium_kwargs)

        self.ground_ill_helper = ground_ill_helper(
            self.ground_constellation, self.iridium_constellation, min_elevation=8.2
        )

        self.add_constellation(self.ground_constellation)
        self.add_constellation(self.iridium_constellation)
        self.add_ill_helper(self.ground_ill_helper)


if __name__ == "__main__":
    iridium = IridiumReconstructedMultiConstellation()
    iridium.update(0.0)
    print(f"Reconstructed Iridium NEXT: {len(iridium.satellites.ids)} nodes in total "
          f"({len(iridium.ground_constellation.satellites.ids)} ground stations and "
          f"{len(iridium.iridium_constellation.satellites.ids)} satellites), "
          f"{len(iridium.isls)} ISLs, {len(iridium.ills)} ILLs")
