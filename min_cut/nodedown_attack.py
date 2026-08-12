"""Minimum-cut satellite removal between two gateways.

The same fixed traffic as the paired baseline, from Fucino to Tempe with no channel loss, plus the
attack itself:

    NodeDownAttackStrategy({9, 26}, start_time=0.0)

Satellites 9 and 26 are the exact minimum node cut between the two stations at the start of the
run, computed in min_cut_analysis.py: they are the only two satellites Fucino can reach at that
instant.

The attack is static. The same set of nodes is held for the whole run, with no tracking and no
retargeting, so its effectiveness varies over time as Fucino hands over from one satellite to the
next. The visibility schedule computed by min_cut_analysis.py gives the three windows this produces.

NodeDownAttackStrategy discards any message whose current hop has one of the given nodes at either
end, which is equivalent to disabling all of that satellite's links at once rather than a single
connection.
"""

import os
import sys
import pickle
from typing import Optional

from dsns.logging import BandwidthLoggingActor, LTPTransmissionLoggingActor, PreprocessedLoggingActor
from dsns.message_actors import BestEffortRoutingDataProvider, LookaheadRoutingDataProvider, MessageRoutingActor
from dsns.simulation import Simulation, LoggingActor
from dsns.message import NodeDownAttackStrategy
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
ATTACKED_NODES = {9, 26}   # min-cut esatto a t=0, vedi topology/min_cut_analysis.py

RESULTS_DIR_DEFAULT = os.path.join(_REPO_ROOT, "results", "min_cut", "under_attack")


def build_node_down_attack(nodes: set, start_time: float, probability: float, seed: Optional[int] = 0):
    return NodeDownAttackStrategy(
        nodes=nodes,
        start_time=start_time,
        probability=probability,
        seed=seed,
        message_filter=None,
    )


def main(
        delivery: str = "best_effort",
        loss: Optional[float] = 0.0,          # no channel loss in this scenario, by design
        message_rate: float = 10.0,           # invariato: coerente con latency_correction.py
        walker_scale: int = 1,
        target_nodes: set = ATTACKED_NODES,
        attack_start_time: float = 0.0,
        attack_probability: float = 1.0,
        verbose: bool = False,
        results_dir: str = RESULTS_DIR_DEFAULT,
):
    if delivery not in ("best_effort", "store_and_forward"):
        raise ValueError(f"Unsupported delivery mode: {delivery} (expected best_effort or store_and_forward)")

    constellation = IridiumReconstructedMultiConstellation(iridium_kwargs=dict(
        num_planes=6,
        sats_per_plane=11 * walker_scale,
    ))
    constellation.update(0.0)

    ground_ids = set(constellation.ground_constellation.satellites.ids)

    lookahead_resolution = 15.0

    actors = []
    data_providers = []

    message_location_tracker = MessageLocationTracker()

    transmission_actor = LinkTransmissionActor(
        default_bandwidth=25e6 // 8,
        buffer_if_link_busy=True,
        reroute_on_link_down=True,
        message_location_tracker=message_location_tracker,
    )
    actors.append(transmission_actor)

    message_config = [("Traffic-0", FUCINO, TEMPE, 1e6, message_rate)]
    traffic_actor = MultiPointToPointTrafficActor(
        message_config=message_config,
        update_interval=60,
        reliable_messages=False,
        cutoff=6000,
    )
    actors.append(traffic_actor)

    loss_config = CustomScenarioLossConfig(seed=0, default_loss_probability=loss, max_frame_size=64 * 1024) if loss else None

    next_hop_override = build_ground_to_ground_override(constellation, ground_ids)

    attack_strategy = build_node_down_attack(
        nodes=set(target_nodes),
        start_time=attack_start_time,
        probability=attack_probability,
    )

    if delivery == "best_effort":
        routing_data_provider = BestEffortRoutingDataProvider(get_next_hop_override=next_hop_override)
        routing_actor = MessageRoutingActor(
            routing_data_provider,
            store_and_forward=False,
            model_bandwidth=True,
            attack_strategy=attack_strategy,
            loss_config=loss_config,
        )
    else:  # store_and_forward
        routing_data_provider = LookaheadRoutingDataProvider(
            resolution=lookahead_resolution,
            num_steps=600,
            get_next_hop_override=next_hop_override,
        )
        routing_actor = MessageRoutingActor(
            routing_data_provider,
            store_and_forward=True,
            model_bandwidth=True,
            attack_strategy=attack_strategy,
            loss_config=loss_config,
        )

    actors.append(routing_actor)
    data_providers.append(routing_data_provider)

    preprocessed_logging_actor = PreprocessedLoggingActor(log_other=False)
    bw_logging_actor = BandwidthLoggingActor()
    ltp_logging_actor = LTPTransmissionLoggingActor()
    logging_actors = [preprocessed_logging_actor, bw_logging_actor, ltp_logging_actor]
    if verbose:
        logging_actors.append(LoggingActor(verbose=True))

    simulation = Simulation(
        constellation,
        actors=actors,
        logging_actors=logging_actors,
        data_providers=data_providers,
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

    nodes_tag = "_".join(str(n) for n in sorted(target_nodes))
    prefix = f"iridium-g2g-eu_usa-nodedown-{delivery}-{loss}-point_to_point-target{FUCINO}_{TEMPE}-nodes{nodes_tag}-{walker_scale}"

    with open(os.path.join(results_dir, f"{prefix}.pickle"), "wb") as f:
        pickle.dump((direct_messages, broadcast_messages, other_events), f)

    with open(os.path.join(results_dir, f"{prefix}-bw-{period}.pickle"), "wb") as f:
        pickle.dump(bw_logging_actor.aggregate(period=period, default_bandwidth=default_bandwidth), f)

    with open(os.path.join(results_dir, f"{prefix}-ltp-{period}.pickle"), "wb") as f:
        pickle.dump(ltp_logging_actor.aggregate(period=period), f)

    print(f"[OK] {prefix} -> {results_dir}")


if __name__ == "__main__":
    main()
