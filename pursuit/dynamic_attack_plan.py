"""Builds the attack plan for the time-aware pursuit.

For each of the visibility windows in which Fucino hands over from one satellite to the next, the
plan records the exact set of satellites to deny: the instantaneous minimum cut for that window,
rather than a fixed set held for the whole run.

The visibility schedule itself is imported from the minimum-cut analysis rather than recomputed:
the constellation, the station and the schedule are identical.

Why the whole visible set is attacked in each window, rather than a single dominant satellite: a
budget-limited variant, holding one satellite at a time and keeping it through the overlaps, was
checked and leaves a large share of the run uncovered, because the cut is frequently larger than
one satellite. The reduced-target trade-off is examined separately, in reduced_target_analysis.py,
on a criterion an external adversary could actually compute.
"""

import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "model_and_baseline"),
    os.path.dirname(os.path.abspath(__file__)),
    os.path.join(_REPO_ROOT, "min_cut"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from presets_reconstructed import IridiumReconstructedMultiConstellation  # noqa: E402
from min_cut_analysis import FUCINO, TEMPE, fucino_ill_schedule  # noqa: E402 - riuso, no duplicazione

SIM_MAX_TIME = 6000.0
TIMESTEP = 15.0


def build_attack_plan(intervals):
    """For each window, targets the whole set of satellites Fucino can reach at that moment.

    This is the instantaneous minimum cut, holding between one and three satellites depending on
    the window.
    """
    plan = [(start, end, sorted(sats)) for start, end, sats in intervals]
    return plan


def plan_stats(plan):
    all_sats = set()
    max_concurrent = 0
    for _, _, sats in plan:
        all_sats.update(sats)
        max_concurrent = max(max_concurrent, len(sats))
    return {
        "n_windows": len(plan),
        "n_retargets": len(plan) - 1,
        "distinct_satellites": sorted(all_sats),
        "n_distinct_satellites": len(all_sats),
        "max_concurrent_targets": max_concurrent,
        "uncovered_time_s": 0.0,   # by construction: the whole visible set is covered in every window
    }


def main():
    constellation = IridiumReconstructedMultiConstellation(iridium_kwargs=dict(num_planes=6, sats_per_plane=11))

    intervals = fucino_ill_schedule(constellation, dt=TIMESTEP, t_max=SIM_MAX_TIME)
    plan = build_attack_plan(intervals)
    stats = plan_stats(plan)

    print(f"Pursuit attack plan: Fucino({FUCINO}) -> Tempe({TEMPE})\n")
    for start, end, sats in plan:
        print(f"  [{start:7.0f} - {end:7.0f}]  dur={end-start:6.0f}s  targets={sats}")

    print(f"\n{stats['n_windows']} finestre, {stats['n_retargets']} cambi di configurazione")
    print(f"Distinct satellites in total: {stats['n_distinct_satellites']} -> {stats['distinct_satellites']}")
    print(f"Peak satellites held simultaneously: {stats['max_concurrent_targets']}")
    print(f"Tempo scoperto: {stats['uncovered_time_s']:.0f}s (0.0%) - per costruzione")

    output = {
        "fucino": FUCINO,
        "tempe": TEMPE,
        "sim_max_time": SIM_MAX_TIME,
        "plan": plan,
        "stats": stats,
    }
    out_path = os.path.join(os.path.dirname(__file__), "dynamic_attack_plan_result.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OK] Plan saved to {out_path}")


if __name__ == "__main__":
    main()
