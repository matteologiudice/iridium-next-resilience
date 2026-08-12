"""Hop-by-hop trace of the Fucino to Tempe traffic, with no attack.

The source and destination pair is fixed rather than drawn at random: Fucino, ground station 5, to
Tempe, ground station 0, the relation this scenario is built around. See min_cut_analysis.py for
the graph analysis that selects those two stations and the satellites targeted later.

Channel loss is set to zero, so that traces are not contaminated by random loss.

The attacked counterpart is message_path_trace_attack.py.
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

# The repository is a set of flat scripts rather than a package, so the shared modules are put on
# the path explicitly.
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

FUCINO = 5   # Fucino/Avezzano, Italia (EU) - sorgente
TEMPE = 0    # Tempe, Arizona, USA: the destination

TRACKED_EVENT_TYPES = (
    MessageCreatedEvent,
    MessageSentEvent,
    MessageReceivedEvent,
    MessageDeliveredEvent,
    MessageDroppedEvent,
)


def main(
        loss: float = 0.0,            # no channel loss in this scenario, by design
        message_rate: float = 10.0,   # invariato: coerente con latency_correction.py (message_rate=10)
        walker_scale: int = 1,
        results_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "res", "message_paths", "normal_traffic"),
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

    # Traffic: a single fixed pair, Fucino to Tempe, rather than randomly chosen ones.
    message_config = [("Traffic-0", FUCINO, TEMPE, 1e6, message_rate)]  # 1 MB every message_rate seconds

    traffic_actor = MultiPointToPointTrafficActor(
        message_config=message_config,
        update_interval=60,
        reliable_messages=False,
        cutoff=6000,
    )

    loss_config = CustomScenarioLossConfig(seed=0, default_loss_probability=loss, max_frame_size=64 * 1024) if loss else None

    next_hop_override = build_ground_to_ground_override(constellation, ground_ids)

    routing_data_provider = BestEffortRoutingDataProvider(get_next_hop_override=next_hop_override)
    routing_actor = MessageRoutingActor(
        routing_data_provider,
        store_and_forward=False,
        model_bandwidth=True,
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
        f"iridium-g2g-eu_usa-best_effort-{loss}-point_to_point-target{FUCINO}_{TEMPE}-{walker_scale}-messagepath.pickle",
    )
    with open(output_file, "wb") as f:
        pickle.dump(message_trace_actor.events, f)

    print(f"[OK] {len(message_trace_actor.events)} eventi tracciati -> {output_file}")


if __name__ == "__main__":
    main()
