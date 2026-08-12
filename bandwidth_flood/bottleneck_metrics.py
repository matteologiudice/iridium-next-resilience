"""Ranks the ground stations by how much of the constellation they see.

A purely geometric analysis, with no simulation involved: for each of the eight reconstructed ground
stations it computes two of the three bottleneck-identification metrics proposed by SKYFALL (Deng
et al., NDSS 2025).

  1. Inter-layer link degree over time, that is, how many satellites each station sees
     simultaneously.
  2. Service time, the accumulated connection time, formalised as the integral over time of that
     degree: for every visibility window, its duration multiplied by the number of simultaneous
     satellites. This follows SKYFALL's own definition and the underlying intuition that a higher
     degree means more traffic passing through, and therefore a more rewarding target.

SKYFALL's third metric, occurrence across geographic blocks, is deliberately not computed. With
eight ground stations, against the far larger global set SKYFALL works with, it would add almost
nothing beyond what the degree and the service time already express. The limit is stated rather
than hidden.
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "model_and_baseline"),
    os.path.dirname(os.path.abspath(__file__)),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ground_stations_reconstructed import IRIDIUM_GROUND_STATIONS_NAMES  # noqa: E402
from presets_base_station import IridiumUserTerminalMultiConstellation  # noqa: E402

SIM_MAX_TIME = 6000.0   # the same horizon used by every other scenario
TIMESTEP = 15.0


def ground_station_ill_schedule(constellation, ground_station_id: int, dt: float = TIMESTEP, t_max: float = SIM_MAX_TIME):
    """Visibility schedule for one ground station: the satellites it sees, interval by interval."""
    schedule = []
    t = 0.0
    while t <= t_max:
        constellation.update(t)
        neighbors = frozenset(
            b if a == ground_station_id else a
            for a, b in constellation.ills if ground_station_id in (a, b)
        )
        schedule.append((t, neighbors))
        t += dt

    intervals = []
    cur_sats, cur_start = schedule[0][1], schedule[0][0]
    for t, sats in schedule[1:]:
        if sats != cur_sats:
            intervals.append((cur_start, t, cur_sats))
            cur_start, cur_sats = t, sats
    intervals.append((cur_start, schedule[-1][0] + dt, cur_sats))
    return intervals


def main():
    constellation = IridiumUserTerminalMultiConstellation(iridium_kwargs=dict(num_planes=6, sats_per_plane=11))

    results = []
    for gs_id in range(len(IRIDIUM_GROUND_STATIONS_NAMES)):
        intervals = ground_station_ill_schedule(constellation, gs_id)

        service_time = sum((end - start) * len(sats) for start, end, sats in intervals)
        concurrent_counts = [len(sats) for _, _, sats in intervals]
        max_concurrent = max(concurrent_counts)
        min_concurrent = min(concurrent_counts)
        avg_concurrent = service_time / SIM_MAX_TIME

        results.append({
            "id": gs_id,
            "name": IRIDIUM_GROUND_STATIONS_NAMES[gs_id].split(" (")[0],
            "service_time": service_time,
            "avg_concurrent": avg_concurrent,
            "max_concurrent": max_concurrent,
            "min_concurrent": min_concurrent,
        })

    print(f"Bottleneck metrics over {SIM_MAX_TIME:.0f}s (occurrence deliberately omitted):\n")
    header = f"{'GS':28s} {'service_time (sat*s)':>22s} {'avg ILL':>9s} {'max ILL':>9s} {'min ILL':>9s}"
    print(header)
    print("-" * len(header))
    for rank_gs in sorted(results, key=lambda r: -r["service_time"]):
        print(f"{rank_gs['name']:28s} {rank_gs['service_time']:22.0f} {rank_gs['avg_concurrent']:9.2f} "
              f"{rank_gs['max_concurrent']:9d} {rank_gs['min_concurrent']:9d}")

    top_service_time = max(results, key=lambda r: r["service_time"])
    top_avg_ill = max(results, key=lambda r: r["avg_concurrent"])
    print(f"\nTop service time: {top_service_time['name']} (ID {top_service_time['id']})")
    print(f"Top grado ILL medio: {top_avg_ill['name']} (ID {top_avg_ill['id']})")
    print(f"Concordanza sulle due metriche: {top_service_time['id'] == top_avg_ill['id']}")


if __name__ == "__main__":
    main()
