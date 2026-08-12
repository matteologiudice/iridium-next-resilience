"""Hop-by-hop trace of the Fucino to Tempe traffic under the minimum-cut attack.

The attacked counterpart of message_path_trace.py, with NodeDownAttackStrategy({9, 26}) active, to
record what actually happens to messages during the three visibility windows identified in
min_cut_analysis.py.

  Blackout    Every satellite Fucino can reach is one of the two attacked, so the graph itself
              guarantees no path exists. An attack drop is expected at the first hop.
  Contested   The attacked satellites are visible alongside at least one that is not, so an
              alternative path exists in principle. Whether routing takes it cannot be deduced from
              the static graph: only an event-by-event trace can answer it.
  Unaffected  Neither attacked satellite is visible, so a path identical to the baseline is
              expected.

Channel loss is zero here as well, matching the baseline trace so the two compare directly.
"""

import os
import sys
import pickle
from typing import Optional

from dsns.simulation import Simulation, LoggingActor
from dsns.message import (
    NodeDownAttackStrategy,
    MessageCreatedEvent,
    MessageSentEvent,
    MessageReceivedEvent,
    MessageDeliveredEvent,
    MessageDroppedEvent,
)
from dsns.message_actors import BestEffortRoutingDataProvider, MessageRoutingActor
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

TRACKED_EVENT_TYPES = (
    MessageCreatedEvent,
    MessageSentEvent,
    MessageReceivedEvent,
    MessageDeliveredEvent,
    MessageDroppedEvent,  # include AttackMessageDroppedEvent (sottoclasse)
)


def build_node_down_attack(nodes: set, start_time: float, probability: float, seed: Optional[int] = 0):
    return NodeDownAttackStrategy(
        nodes=nodes,
        start_time=start_time,
        probability=probability,
        seed=seed,
        message_filter=None,
    )


def main(
        loss: float = 0.0,             # no channel loss in this scenario, by design
        message_rate: float = 10.0,    # invariato: coerente con latency_correction.py
        walker_scale: int = 1,
        target_nodes: set = ATTACKED_NODES,
        attack_start_time: float = 0.0,
        attack_probability: float = 1.0,
        results_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "res", "message_paths", "under_attack"),
):
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

    attack_strategy = build_node_down_attack(
        nodes=set(target_nodes),
        start_time=attack_start_time,
        probability=attack_probability,
    )

    routing_data_provider = BestEffortRoutingDataProvider(get_next_hop_override=next_hop_override)
    routing_actor = MessageRoutingActor(
        routing_data_provider,
        store_and_forward=False,
        model_bandwidth=True,
        attack_strategy=attack_strategy,
        loss_config=loss_config,
    )

    message_trace_actor = LoggingActor(
        event_filter=lambda event: isinstance(event, TRACKED_EVENT_TYPES)
    )

    simulation = Simulation(
        constellation,
        actors=[transmission_actor, traffic_actor, routing_actor],
        logging_actors=[message_trace_actor],
        data_providers=[routing_data_provider],
        timestep=15.0,
    )

    simulation.initialize(time=0)
    simulation.run(6000 * 4, progress=True)

    os.makedirs(results_dir, exist_ok=True)
    nodes_tag = "_".join(str(n) for n in sorted(target_nodes))
    output_file = os.path.join(
        results_dir,
        f"iridium-g2g-eu_usa-nodedown-messagepath-best_effort-{loss}-point_to_point-target{FUCINO}_{TEMPE}-nodes{nodes_tag}-{walker_scale}.pickle",
    )
    with open(output_file, "wb") as f:
        pickle.dump(message_trace_actor.events, f)

    print(f"[OK] {len(message_trace_actor.events)} eventi tracciati -> {output_file}")


if __name__ == "__main__":
    main()
