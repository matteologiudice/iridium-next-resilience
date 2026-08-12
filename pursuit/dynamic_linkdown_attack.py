"""Time-aware pursuit of a ground station's uplink.

The same fixed traffic as the static scenario, from Fucino to Tempe with no channel loss, but with
an attack that moves. Instead of holding one set of satellites for the whole run, it follows which
satellites Fucino is actually using, window by window, taking the instantaneous minimum cut from the
plan built by dynamic_attack_plan.py. That cut holds between one and three satellites depending on
the window, the larger sets occurring during handovers.

Why a custom function rather than the simulator's own strategies
----------------------------------------------------------------
The provided attack strategies take a fixed set of links or nodes and a start time, and once active
they persist for the rest of the run. Composing one per window with MultipleAttackStrategy, which
combines strategies by logical disjunction, would accumulate targets instead of replacing them: by
the end of the run the adversary would be holding down every satellite it had ever targeted, a
different and far larger attack than the one intended.

The strategy used here is a plain function that looks the current time up in the plan at every
message-sent event and consults only the window in force at that instant, carrying no state between
windows. Nothing in the simulator had to be modified: an attack strategy is a type alias for any
function taking a send event and returning a boolean, rather than a class to be subclassed.

The strategy denies only the hop between Fucino and the targeted satellite, leaving every other
link of that satellite intact.
"""

import json
import os
import sys
import pickle
from typing import Optional

from dsns.logging import BandwidthLoggingActor, LTPTransmissionLoggingActor, PreprocessedLoggingActor
from dsns.message_actors import BestEffortRoutingDataProvider, LookaheadRoutingDataProvider, MessageRoutingActor
from dsns.simulation import Simulation, LoggingActor
from dsns.message import MessageSentEvent
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

PLAN_PATH = os.path.join(_REPO_ROOT, "results", "pursuit", "dynamic_attack_plan_result.json")
RESULTS_DIR_DEFAULT = os.path.join(_REPO_ROOT, "results", "pursuit", "under_attack")


def load_attack_plan(plan_path: str = PLAN_PATH):
    with open(plan_path) as f:
        data = json.load(f)
    # Each entry is (start, end, [satellites]), with the satellite list held as a frozenset
    return [(s, e, frozenset(sats)) for s, e, sats in data["plan"]]


def build_dynamic_linkdown_attack(plan, fucino: int = FUCINO, probability: float = 1.0, seed: Optional[int] = 0):
    """Custom strategy matching the AttackStrategy protocol, Callable[[MessageSentEvent], bool].

    For every hop it finds which window of the plan the event's timestamp falls into, then denies
    the hop only if it runs between Fucino and one of the satellites targeted in that window. This
    is LinkDown semantics: never the whole node, only the specific edge to or from Fucino.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    def strategy(event: MessageSentEvent) -> bool:
        for start, end, sats in plan:
            if start <= event.time < end:
                is_fucino_hop = (
                    (event.source == fucino and event.destination in sats)
                    or (event.destination == fucino and event.source in sats)
                )
                return is_fucino_hop and rng.random() < probability
        return False

    return strategy


def main(
        delivery: str = "best_effort",
        loss: Optional[float] = 0.0,          # no channel loss in this scenario, by design
        message_rate: float = 10.0,           # invariato: coerente con latency_correction.py
        walker_scale: int = 1,
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

    plan = load_attack_plan()
    attack_strategy = build_dynamic_linkdown_attack(plan, probability=attack_probability)

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

    prefix = f"iridium-g2g-eu_usa-dynamiclinkdown-{delivery}-{loss}-point_to_point-target{FUCINO}_{TEMPE}-{walker_scale}"

    with open(os.path.join(results_dir, f"{prefix}.pickle"), "wb") as f:
        pickle.dump((direct_messages, broadcast_messages, other_events), f)

    with open(os.path.join(results_dir, f"{prefix}-bw-{period}.pickle"), "wb") as f:
        pickle.dump(bw_logging_actor.aggregate(period=period, default_bandwidth=default_bandwidth), f)

    with open(os.path.join(results_dir, f"{prefix}-ltp-{period}.pickle"), "wb") as f:
        pickle.dump(ltp_logging_actor.aggregate(period=period), f)

    print(f"[OK] {prefix} -> {results_dir}")


if __name__ == "__main__":
    main()
