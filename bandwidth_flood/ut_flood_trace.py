"""Hop-by-hop trace of individual messages during the bandwidth exhaustion attack.

The tracing counterpart of ut_flood_attack.py, with the same topology, the same twenty-four
terminals, the same dimensioning and the same attack window. Where the attack script records
aggregate statistics, this one records the sequence of nodes each message traverses, for both the
legitimate flows into Svalbard and the messages the compromised terminals generate.
"""

import os
import sys
import pickle

from dsns.message import (
    MessageCreatedEvent,
    MessageSentEvent,
    MessageReceivedEvent,
    MessageDeliveredEvent,
    MessageDroppedEvent,
)
from dsns.message_actors import BestEffortRoutingDataProvider, MessageRoutingActor
from dsns.simulation import Simulation, LoggingActor
from dsns.traffic_sim import MultiPointToPointTrafficActor
from dsns.transmission import MessageLocationTracker, LinkTransmissionActor

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "model_and_baseline"),
    os.path.dirname(os.path.abspath(__file__)),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from presets_base_station import IridiumUserTerminalMultiConstellation  # noqa: E402
from iridium_ground_to_ground import build_ground_to_ground_override  # noqa: E402
from flood_actor import SimpleFloodActor  # noqa: E402

FUCINO = 5
TEMPE = 0
SVALBARD = 3

FLOOD_START = 990.0
FLOOD_END = 1080.0
FLOOD_SIZE = 1_000_000
FLOOD_PERIOD = 2.0   # one message every two seconds per terminal, that is 4 Mbps each

TRACKED_EVENT_TYPES = (
    MessageCreatedEvent,
    MessageSentEvent,
    MessageReceivedEvent,
    MessageDeliveredEvent,
    MessageDroppedEvent,
)

RESULTS_DIR_DEFAULT = os.path.join(_REPO_ROOT, "results", "bandwidth_flood", "message_paths")


def main(message_rate: float = 10.0, walker_scale: int = 1, results_dir: str = RESULTS_DIR_DEFAULT):
    constellation = IridiumUserTerminalMultiConstellation(iridium_kwargs=dict(
        num_planes=6,
        sats_per_plane=11 * walker_scale,
    ))
    constellation.update(0.0)

    gs_ids = set(constellation.ground_constellation.satellites.ids)
    ut_ids = sorted(constellation.ut_constellation.satellites.ids)
    ground_ids = gs_ids | set(ut_ids)
    compromised_uts = ut_ids

    message_location_tracker = MessageLocationTracker()

    transmission_actor = LinkTransmissionActor(
        default_bandwidth=25e6 // 8,
        buffer_if_link_busy=True,
        reroute_on_link_down=True,
        message_location_tracker=message_location_tracker,
    )

    message_config = [
        ("Traffic-0", FUCINO, SVALBARD, 1e6, message_rate),
        ("Traffic-1", TEMPE, SVALBARD, 1e6, message_rate),
    ]
    traffic_actor = MultiPointToPointTrafficActor(
        message_config=message_config,
        update_interval=60,
        reliable_messages=False,
        cutoff=6000,
    )

    flood_links = [(ut, SVALBARD) for ut in compromised_uts]
    flood_actor = SimpleFloodActor(
        links=flood_links,
        period=FLOOD_PERIOD,
        size=FLOOD_SIZE,
        start_time=FLOOD_START,
        end_time=FLOOD_END,
    )

    next_hop_override = build_ground_to_ground_override(constellation, ground_ids)
    routing_data_provider = BestEffortRoutingDataProvider(get_next_hop_override=next_hop_override)
    routing_actor = MessageRoutingActor(
        routing_data_provider,
        store_and_forward=False,
        model_bandwidth=True,
    )

    message_trace_actor = LoggingActor(
        event_filter=lambda event: isinstance(event, TRACKED_EVENT_TYPES)
    )

    simulation = Simulation(
        constellation,
        actors=[transmission_actor, traffic_actor, flood_actor, routing_actor],
        logging_actors=[message_trace_actor],
        data_providers=[routing_data_provider],
        timestep=15.0,
    )

    simulation.initialize(time=0)
    simulation.run(6000 * 4, progress=True)

    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, "ut-flood-messagepath-best_effort.pickle")
    with open(output_file, "wb") as f:
        pickle.dump(message_trace_actor.events, f)

    print(f"[OK] {len(message_trace_actor.events)} eventi tracciati -> {output_file}")


if __name__ == "__main__":
    main()
