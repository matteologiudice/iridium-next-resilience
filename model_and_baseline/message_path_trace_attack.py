"""Hop-by-hop trace of individual messages while a link is under attack.

The attacked counterpart of message_path_trace.py: same generic LoggingActor and same event filter
over the message travel events, but with a LinkDownAttackStrategy passed to MessageRoutingActor.
Default parameters match the baseline trace, so the same seeds produce the same source and
destination pairs and the two runs can be compared message by message without regenerating the
baseline.

The aggregate attack run uses the three summarising logging actors instead, which report how many
messages were dropped but not the sequence of nodes each one traversed. That is why this script
exists separately.

How the attack appears in a trace
---------------------------------
MessageRoutingActor evaluates the attack strategy after the MessageSentEvent for a hop has already
been generated. When the attack fires, the actor returns an AttackMessageDroppedEvent instead of
continuing to a MessageReceivedEvent. A traced path is therefore identical to its baseline
counterpart up to and including the attacked hop, and then simply stops: no later hop is ever
recorded. Routing, including the ISL-only override, never learns that the link is unavailable, so
no rerouting should be expected.

AttackMessageDroppedEvent inherits from MessageDroppedEvent but carries no explicit reason, so its
reason field stays at the default of DropReason.UNKNOWN rather than the DropReason.NO_NEXT_HOP that
marks a structural drop. The generic LoggingActor records it correctly and no correction is needed.

Default target
--------------
Link (26, 27), the busiest inter-satellite link in this scenario, identified from the baseline
bandwidth records. With traffic kept light for tracing, there is no guarantee that either traced
message crosses that particular link; the script prints each message's full path so this can be
checked directly, and --traffic-scale or the seed can be raised if neither does. The two comparison
targets remain available through --target-link.

Configuration
-------------
Constellation, traffic and routing are those of message_path_trace.py: the eight reconstructed
ground stations, endpoints drawn from ground stations alone, and BestEffortRoutingDataProvider
receiving both get_next_hop_override and the attack strategy. Output is a single pickle file under
results/message_paths with the "iridium-g2g-linkdown-messagepath" prefix.
"""

import os
import sys
import pickle
import random
import argparse
from typing import Optional

from dsns.simulation import Simulation, LoggingActor
from dsns.message import (
    LinkDownAttackStrategy,
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
# the path explicitly: the repository root for presets_reconstructed, and model_and_baseline for
# iridium_ground_to_ground, from which CustomScenarioLossConfig and build_ground_to_ground_override
# are imported rather than duplicated.
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

# Busiest inter-satellite link in this scenario. Both satellites carry the full degree of four,
# so this is the most heavily loaded link rather than a structurally weak one.
DEFAULT_TARGET_LINK = (26, 27)

# The two comparison targets, available through --target-link.
LOW_TRAFFIC_TARGET_LINK = (35, 34)     # 1 MB over a single active interval
ZERO_TRAFFIC_TARGET_LINK = (70, 71)    # never traversed by any message (negative control)

RESULTS_DIR_DEFAULT = os.path.join(_REPO_ROOT, "results", "message_paths")


# The events needed to reconstruct a message's path, as in the baseline trace.
TRACKED_EVENT_TYPES = (
    MessageCreatedEvent,
    MessageSentEvent,
    MessageReceivedEvent,
    MessageDeliveredEvent,
    MessageDroppedEvent,  # include AttackMessageDroppedEvent (sottoclasse)
)


def build_link_down_attack(target_link: tuple[int, int], start_time: float, probability: float, seed: Optional[int] = 0):
    """Builds the attack strategy, supplying both orderings of the target pair."""
    a, b = target_link
    links = {(a, b), (b, a)}
    return LinkDownAttackStrategy(
        links=links,
        start_time=start_time,
        probability=probability,
        seed=seed,
        message_filter=None,
    )


def main(
        traffic_scale: int = 2,          # same default as the baseline trace, so the two align
        loss: float = 0.005,              # same default as the baseline trace
        walker_scale: int = 1,
        target_link: tuple[int, int] = DEFAULT_TARGET_LINK,
        attack_start_time: float = 0.0,
        attack_probability: float = 1.0,
        results_dir: str = RESULTS_DIR_DEFAULT,
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

    # Ground-to-ground traffic identical to the baseline trace: the same seed per index yields the
    # same source and destination pairs, so the two runs compare message by message.
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

    # The ground-to-ground, ISL-only constraint, identical to every other run.
    next_hop_override = build_ground_to_ground_override(constellation, ground_ids)

    attack_strategy = build_link_down_attack(
        target_link=target_link,
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
    a, b = target_link
    output_file = os.path.join(
        results_dir,
        f"iridium-g2g-linkdown-messagepath-best_effort-{loss}-point_to_point-{traffic_scale}-{walker_scale}"
        f"-target{a}_{b}-start{attack_start_time}-prob{attack_probability}.pickle",
    )
    with open(output_file, "wb") as f:
        pickle.dump(message_trace_actor.events, f)

    print(f"[OK] {len(message_trace_actor.events)} eventi tracciati -> {output_file}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hop-by-hop trace of Iridium NEXT messages under LinkDownAttackStrategy, on "
                    "the ground-to-ground, ISL-only model with eight reconstructed ground stations."
    )
    parser.add_argument("--traffic-scale", type=int, default=2)
    parser.add_argument("--loss", type=float, default=0.005)
    parser.add_argument("--walker-scale", type=int, default=1)
    parser.add_argument("--target-link", type=str, default=f"{DEFAULT_TARGET_LINK[0]},{DEFAULT_TARGET_LINK[1]}",
                         help="Pair of satellite IDs to attack, for example '26,27' (default: the busiest "
                              "link). Also available: '35,34' (lightly loaded), '70,71' (never traversed).")
    parser.add_argument("--attack-start-time", type=float, default=0.0)
    parser.add_argument("--attack-probability", type=float, default=1.0)
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR_DEFAULT)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    a_str, b_str = args.target_link.split(",")
    target_link = (int(a_str.strip()), int(b_str.strip()))

    main(
        traffic_scale=args.traffic_scale,
        loss=args.loss,
        walker_scale=args.walker_scale,
        target_link=target_link,
        attack_start_time=args.attack_start_time,
        attack_probability=args.attack_probability,
        results_dir=args.results_dir,
    )
