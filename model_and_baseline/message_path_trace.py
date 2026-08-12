"""Hop-by-hop trace of individual messages on the baseline model.

Answers a different question from the aggregate runs: not how many messages arrived, but how a
single message travels, node by node, from its source to its destination. It also serves as the
direct check that the ISL-only constraint holds, since every recorded path should read

    ground station -> satellite -> satellite -> ... -> satellite -> ground station

with no intermediate hop passing through a ground station.

The three aggregate logging actors (PreprocessedLoggingActor, BandwidthLoggingActor,
LTPTransmissionLoggingActor) are deliberately not used here: they report statistics that are already
summarised, such as a hop count, rather than the sequence of nodes actually traversed. This script
uses the generic dsns.simulation.LoggingActor instead, with an event filter that keeps the
create, send, receive, deliver and drop events for each message.

Configuration
-------------
The constellation is IridiumReconstructedMultiConstellation, with the eight reconstructed ground
stations in place of the 256 synthetic defaults. Source and destination pairs are drawn from the
ground stations alone, never from satellites. Traffic is kept deliberately light, two flows by
default, so that individual paths can be followed without noise. Routing uses
BestEffortRoutingDataProvider with the override built by build_ground_to_ground_override(),
imported from the baseline rather than duplicated, so the routing behaviour is identical to the
runs it is meant to explain.

Output is a single pickle file under results/message_paths, carrying the same "iridium-g2g" prefix
as the aggregate runs.
"""

import os
import sys
import pickle
import random

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
# the path explicitly: the repository root, and model_and_baseline for
# iridium_ground_to_ground.py, da cui importiamo CustomScenarioLossConfig e
# build_ground_to_ground_override, rather than duplicating them.
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


# The events needed to reconstruct a message's path. Everything else, such as connectivity
# updates and transmission queue activity, is discarded by the filter.
TRACKED_EVENT_TYPES = (
    MessageCreatedEvent,
    MessageSentEvent,
    MessageReceivedEvent,
    MessageDeliveredEvent,
    MessageDroppedEvent,
)


def main(
        traffic_scale: int = 2,       # deliberately few flows, so individual messages can be followed
        loss: float = 0.005,          # 0.5%, the same setting used by the aggregate runs
        walker_scale: int = 1,
        results_dir: str = os.path.join(_REPO_ROOT, "results", "message_paths"),
):
    constellation = IridiumReconstructedMultiConstellation(iridium_kwargs=dict(
        num_planes=6,
        sats_per_plane=11 * walker_scale,
    ))
    constellation.update(0.0)  # populates .isls/.ills/.satellites before the override is built

    ground_ids = set(constellation.ground_constellation.satellites.ids)
    ground_ids_list = list(ground_ids)

    message_location_tracker = MessageLocationTracker()

    transmission_actor = LinkTransmissionActor(
        default_bandwidth=25e6 // 8,
        buffer_if_link_busy=True,
        reroute_on_link_down=True,
        message_location_tracker=message_location_tracker,
    )

    # Ground-to-ground traffic: source and destination pairs are drawn from the eight
    # reconstructed ground stations alone, never from the satellites.
    message_config = []
    for i in range(traffic_scale):
        random.seed(i)
        source, destination = random.sample(ground_ids_list, 2)
        message_config.append((f"Traffic-{i}", source, destination, 1e6, 10.0))  # 1 MB every 10 s

    traffic_actor = MultiPointToPointTrafficActor(
        message_config=message_config,
        update_interval=60,
        reliable_messages=False,
        cutoff=6000,
    )

    loss_config = CustomScenarioLossConfig(seed=0, default_loss_probability=loss, max_frame_size=64 * 1024) if loss else None

    # The ground-to-ground, ISL-only constraint, identical to the one used by the aggregate runs.
    next_hop_override = build_ground_to_ground_override(constellation, ground_ids)

    routing_data_provider = BestEffortRoutingDataProvider(get_next_hop_override=next_hop_override)
    routing_actor = MessageRoutingActor(
        routing_data_provider,
        store_and_forward=False,
        model_bandwidth=True,
        loss_config=loss_config,
    )

    # The only logging actor used here: it records the message travel events listed in
    # TRACKED_EVENT_TYPES above and discards everything else.
    message_trace_actor = LoggingActor(
        event_filter=lambda event: isinstance(event, TRACKED_EVENT_TYPES)
    )

    simulation = Simulation(
        constellation,
        actors=[transmission_actor, traffic_actor, routing_actor],
        logging_actors=[message_trace_actor],  # no Preprocessed/Bandwidth/LTP actor, only the trace
        data_providers=[routing_data_provider],
        timestep=15.0,
    )

    simulation.initialize(time=0)
    simulation.run(6000 * 4, progress=True)

    # --- Output: a single pickle file in a dedicated subdirectory -------------------------------
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(
        results_dir,
        f"iridium-g2g-best_effort-{loss}-point_to_point-{traffic_scale}-{walker_scale}-messagepath.pickle",
    )
    with open(output_file, "wb") as f:
        pickle.dump(message_trace_actor.events, f)

    print(f"[OK] {len(message_trace_actor.events)} eventi tracciati -> {output_file}")


if __name__ == "__main__":
    main()
