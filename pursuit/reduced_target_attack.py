"""Empirical check of the reduced-target prediction.

Runs the simulation against the K most visible satellites, chosen on the externally observable
criterion rather than on real traffic volumes, to verify the coverage that reduced_target_analysis.py
predicts by counting.

Unlike the full pursuit, no time windows and no custom strategy are needed here: the target set is
fixed for the whole run, attacked whenever Fucino uses one of its members, so the simulator's own
LinkDownAttackStrategy is sufficient with a start time of zero.

Run as a separate Python process: PreprocessedLoggingActor keeps state across calls to main()
within a single process.
"""

import json
import os
import sys
import pickle
from typing import Optional

from dsns.logging import BandwidthLoggingActor, LTPTransmissionLoggingActor, PreprocessedLoggingActor
from dsns.message_actors import BestEffortRoutingDataProvider, MessageRoutingActor
from dsns.simulation import Simulation, LoggingActor
from dsns.message import LinkDownAttackStrategy
from dsns.transmission import MessageLocationTracker, LinkTransmissionActor
from dsns.traffic_sim import MultiPointToPointTrafficActor

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "model_and_baseline"),
    os.path.dirname(os.path.abspath(__file__)),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from presets_reconstructed import IridiumReconstructedMultiConstellation  # noqa: E402
from iridium_ground_to_ground import CustomScenarioLossConfig, build_ground_to_ground_override  # noqa: E402

FUCINO = 5
TEMPE = 0

REDUCED_ANALYSIS_PATH = os.path.join(_REPO_ROOT, "results", "pursuit", "reduced_target_analysis_result.json")
RESULTS_DIR_DEFAULT = os.path.join(_REPO_ROOT, "results", "pursuit", "reduced_target")


def load_target_satellites(k: int = 14, path: str = REDUCED_ANALYSIS_PATH):
    with open(path) as f:
        data = json.load(f)
    return sorted(data["ranking_by_visibility"][:k])


def build_reduced_target_attack(target_satellites, fucino: int = FUCINO):
    """Builds the attack from the simulator's own LinkDownAttackStrategy.

    No custom function is needed here: the set of Fucino to satellite links is static, active from
    the start of the run, and supplied in both directions.
    """
    links = set()
    for sat in target_satellites:
        links.add((fucino, sat))
        links.add((sat, fucino))
    return LinkDownAttackStrategy(links=links, start_time=0.0, probability=1.0, seed=0)


def main(
        delivery: str = "best_effort",
        k: int = 14,
        loss: Optional[float] = 0.0,
        message_rate: float = 10.0,
        walker_scale: int = 1,
        verbose: bool = False,
        results_dir: str = RESULTS_DIR_DEFAULT,
):
    if delivery != "best_effort":
        raise ValueError("This script checks a prediction made on the best-effort baseline trace: "
                         "use best_effort for the comparison to be valid")

    target_satellites = load_target_satellites(k=k)
    print(f"K={k} target satellites: {target_satellites}")

    constellation = IridiumReconstructedMultiConstellation(iridium_kwargs=dict(
        num_planes=6,
        sats_per_plane=11 * walker_scale,
    ))
    constellation.update(0.0)

    ground_ids = set(constellation.ground_constellation.satellites.ids)

    message_location_tracker = MessageLocationTracker()

    transmission_actor = LinkTransmissionActor(
        default_bandwidth=25e6 // 8,
        buffer_if_link_busy=True,
        reroute_on_link_down=True,
        message_location_tracker=message_location_tracker,
    )

    message_config = [("Traffic-0", FUCINO, TEMPE, 1e6, message_rate)]
    traffic_actor = MultiPointToPointTrafficActor(
        message_config=message_config,
        update_interval=60,
        reliable_messages=False,
        cutoff=6000,
    )

    loss_config = CustomScenarioLossConfig(seed=0, default_loss_probability=loss, max_frame_size=64 * 1024) if loss else None

    next_hop_override = build_ground_to_ground_override(constellation, ground_ids)

    attack_strategy = build_reduced_target_attack(target_satellites)

    routing_data_provider = BestEffortRoutingDataProvider(get_next_hop_override=next_hop_override)
    routing_actor = MessageRoutingActor(
        routing_data_provider,
        store_and_forward=False,
        model_bandwidth=True,
        attack_strategy=attack_strategy,
        loss_config=loss_config,
    )

    preprocessed_logging_actor = PreprocessedLoggingActor(log_other=False)
    bw_logging_actor = BandwidthLoggingActor()
    ltp_logging_actor = LTPTransmissionLoggingActor()
    logging_actors = [preprocessed_logging_actor, bw_logging_actor, ltp_logging_actor]
    if verbose:
        logging_actors.append(LoggingActor(verbose=True))

    simulation = Simulation(
        constellation,
        actors=[transmission_actor, traffic_actor, routing_actor],
        logging_actors=logging_actors,
        data_providers=[routing_data_provider],
        timestep=15.0,
    )

    simulation.initialize(time=0)
    simulation.run(6000 * 4, progress=True)

    os.makedirs(results_dir, exist_ok=True)

    direct_messages = preprocessed_logging_actor.direct_messages
    broadcast_messages = preprocessed_logging_actor.broadcast_messages
    other_events = preprocessed_logging_actor.other_events

    default_bandwidth = transmission_actor._default_bandwidth
    period = 1.0

    sats_tag = "_".join(str(s) for s in target_satellites)
    prefix = f"iridium-g2g-eu_usa-reducedtarget-k{k}-{delivery}-{loss}-point_to_point-target{FUCINO}_{TEMPE}-sats{sats_tag}-1"

    with open(os.path.join(results_dir, f"{prefix}.pickle"), "wb") as f:
        pickle.dump((direct_messages, broadcast_messages, other_events), f)

    with open(os.path.join(results_dir, f"{prefix}-bw-{period}.pickle"), "wb") as f:
        pickle.dump(bw_logging_actor.aggregate(period=period, default_bandwidth=default_bandwidth), f)

    with open(os.path.join(results_dir, f"{prefix}-ltp-{period}.pickle"), "wb") as f:
        pickle.dump(ltp_logging_actor.aggregate(period=period), f)

    # Statistics recomputed by reading back the pickle just written, so the file on disk is the
    # single source of truth.
    with open(os.path.join(results_dir, f"{prefix}.pickle"), "rb") as f:
        saved_direct_messages, _, _ = pickle.load(f)
    msgs = list(saved_direct_messages.values())
    n = len(msgs)
    delivered = sum(1 for m in msgs if m.delivered)
    dropped = sum(1 for m in msgs if m.dropped)
    limbo = n - delivered - dropped
    print(f"\nMEASURED RESULT (read back from disk): n={n}  delivered={delivered} ({100*delivered/n:.1f}%)  "
          f"dropped={dropped} ({100*dropped/n:.1f}%)  limbo={limbo} ({100*limbo/n:.1f}%)")
    print(f"Drop rate osservato: {100*dropped/n:.1f}%  (previsto teoricamente: 84.3%)")

    print(f"\n[OK] {prefix} -> {results_dir}")


if __name__ == "__main__":
    main()
