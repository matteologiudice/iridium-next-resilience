# Iridium NEXT resilience under targeted attack

Simulation code accompanying the master's thesis *Attacking Iridium NEXT: A Simulation-Based Study
of LEO Constellation Resilience under Targeted Cyber Attacks* (Matteo Lo Giudice, Università
Bocconi and Politecnico di Milano).

The repository contains the model of the Iridium NEXT constellation used in the thesis, the
reconstruction of its ground segment, the baseline traffic simulation, and the four attack
scenarios examined in Chapter 3. It is built on the [Deep Space Network
Simulator](https://github.com/ssloxford/dsns) and modifies none of its source.

## What is modelled

Sixty-six satellites in six near-polar planes at 780 km, each linked to its neighbours ahead and
behind within its own plane and to the corresponding satellites in the adjacent planes, with
inter-plane links disabled across the seam. The ground segment is a reconstruction of Iridium
NEXT's eight real sites, assembled from FCC filings, SEC filings, corporate disclosures and
independent reporting; per-site provenance and coordinate confidence are documented in
`ground_stations_reconstructed.py`.

Routing is constrained so that message sources and destinations are always ground stations and
every intermediate hop is a satellite reached over an inter-satellite link, which the simulator does
not enforce by default.

## Where to find what

| Thesis | Purpose | File |
|---|---|---|
| § 2.3.2 | Reconstructed ground station coordinates | `ground_stations_reconstructed.py` |
| § 2.3.1 | Constellation and ground segment composition | `presets_reconstructed.py` |
| § 2.3.3, § 2.4 | ISL-only routing constraint, baseline run | `model_and_baseline/iridium_ground_to_ground.py` |
| § 3.2 | Disabling a single inter-satellite link | `link_down/linkdown_attack.py` |
| § 3.3.2 | Minimum-cut graph analysis and visibility windows | `min_cut/min_cut_analysis.py` |
| § 3.3 | Paired baseline, Fucino to Tempe | `min_cut/eu_usa_baseline.py` |
| § 3.3 | Minimum-cut satellite removal | `min_cut/nodedown_attack.py` |
| § 3.4.2 | Attack plan for the time-aware pursuit | `pursuit/dynamic_attack_plan.py` |
| § 3.4 | Time-aware pursuit of a ground station's uplink | `pursuit/dynamic_linkdown_attack.py` |
| § 3.4.4 | Reduced-target coverage analysis | `pursuit/reduced_target_analysis.py` |
| § 3.4.4 | Empirical check of the reduced-target prediction | `pursuit/reduced_target_attack.py` |
| § 3.5.2 | Synthetic user terminals across three regions | `bandwidth_flood/ut_synthetic_multiregion.py` |
| § 3.5.2 | Ground station ranking by service time and link degree | `bandwidth_flood/bottleneck_metrics.py` |
| § 3.5.2 | Extended topology with user terminals | `bandwidth_flood/presets_base_station.py` |
| § 3.5 | Paired baseline with terminals present but silent | `bandwidth_flood/ut_scenario_baseline.py` |
| § 3.5 | Distributed bandwidth exhaustion | `bandwidth_flood/ut_flood_attack.py` |

The `message_path_trace*.py` scripts in each folder record hop-by-hop paths rather than aggregate
statistics, and the `visualize_*.py` scripts produce the 3D renderings that appear as figures.

## Running it

```
git clone https://github.com/ssloxford/dsns
pip install -e dsns
pip install -r requirements.txt
```

Each script is standalone and takes its parameters from the command line:

```
python3 model_and_baseline/iridium_ground_to_ground.py --delivery best_effort --loss 0.0
python3 link_down/linkdown_attack.py --target-link 26,27
python3 min_cut/min_cut_analysis.py
python3 pursuit/dynamic_attack_plan.py
python3 bandwidth_flood/ut_flood_attack.py --delivery store_and_forward
```

Results are written under `results/`, which is not tracked.

Two practical constraints. Every configuration must be run as a separate Python process:
`PreprocessedLoggingActor` keeps state across calls to `main()`, so two runs in one process
contaminate each other. And the 3D rendering scripts require Python 3.11 with pyglet 1.5, as pinned
in `requirements.txt`; under pyglet 2.x the offscreen renderer produces cropped or empty images.

## Reproducibility

Every result in the thesis comes from a single execution of its configuration. There are no averages
over repeated runs and no confidence intervals, because the sources of variation are fixed in
advance: the order in which the simulation processes its components at each event is deterministic,
ground station pairs come from a generator initialised with a fixed seed, and the loss model draws
from a per-link generator seeded once at the start of the run.

The reported figures were produced against DSNS at commit
`7e3e4f045ae3183aeacfce41b65ea9e275b9d29f`.

## Licence and attribution

Released under the GNU General Public License v3.0, the licence of the simulator this work builds
on. See `LICENSE`.

The Deep Space Network Simulator is the work of Smailes, Futera, Köhler, Birnbach, Strohmeier and
Martinovic, presented at *2025 Security for Space Systems (3S)*. No file of theirs is modified here:
this repository composes their public classes and functions with different arguments, and uses the
extension points they expose.
