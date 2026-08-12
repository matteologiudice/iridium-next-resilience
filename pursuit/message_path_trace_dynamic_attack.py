"""Hop-by-hop trace of the Fucino to Tempe traffic under the pursuit attack.

Traces what actually happens to messages while the attack follows the target as the constellation
moves, using the same custom strategy as the aggregate run, imported rather than duplicated.

Every message is expected to stop at its first hop, from Fucino to whichever satellite it is
trying to reach, because the plan covers every instant of the run and the strategy always denies
the uplink in force at that moment. This is the contrast with the static scenario, where messages
that never touched the two attacked satellites passed undisturbed.
"""

import os
import sys
import pickle

from dsns.simulation import Simulation, LoggingActor
from dsns.message import (
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
from dynamic_linkdown_attack import load_attack_plan, build_dynamic_linkdown_attack  # noqa: E402 - riuso, no duplicazione

FUCINO = 5
TEMPE = 0

TRACKED_EVENT_TYPES = (
    MessageCreatedEvent,
    MessageSentEvent,
    MessageReceivedEvent,
    MessageDeliveredEvent,
    MessageDroppedEvent,  # include AttackMessageDroppedEvent (sottoclasse)
)


def main(
        loss: float = 0.0,
        message_rate: float = 10.0,
        walker_scale: int = 1,
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

    plan = load_attack_plan()
    attack_strategy = build_dynamic_linkdown_attack(plan, probability=attack_probability)

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
    output_file = os.path.join(
        results_dir,
        f"iridium-g2g-eu_usa-dynamiclinkdown-messagepath-best_effort-{loss}-point_to_point-target{FUCINO}_{TEMPE}-{walker_scale}.pickle",
    )
    with open(output_file, "wb") as f:
        pickle.dump(message_trace_actor.events, f)

    print(f"[OK] {len(message_trace_actor.events)} eventi tracciati -> {output_file}")


if __name__ == "__main__":
    main()
