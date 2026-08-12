"""Disabling a single inter-satellite link.

Runs the LinkDown attack on the reconstructed Iridium NEXT model, using the same ground-to-ground,
ISL-only configuration as the baseline, with one inter-satellite link denied for the whole run.

Target selection
----------------
The links below were identified from the baseline's own bandwidth records rather than from the
topology. For every directed pair of nodes carrying traffic at any point in a run, the logging
component records the volume transmitted and the number of one-second intervals in which
transmission was active. Ranking those pairs by volume orders the network by how heavily the
routing layer loads each connection. The ranking was computed separately under both loss settings
and agrees on the busiest link.

  (26, 27)  Busiest link. Between 408 and 537 MB transmitted depending on the loss setting, over
            520 to 525 active intervals, ranking first or second under both. Both satellites carry
            the full degree of four, so this is a target chosen for observed load rather than for
            structural fragility.
  (35, 34)  Lightly loaded link. 1 MB over a single active interval, the smallest non-zero value
            found.
  (70, 71)  Never traversed. Present among the constellation's 121 inter-satellite links but
            carrying no traffic in either run. Serves as a negative control: an attack on it should
            produce no AttackMessageDroppedEvent at all.

Mechanism
---------
dsns.message.LinkDownAttackStrategy is passed to MessageRoutingActor through attack_strategy. It is
evaluated once a message has been routed and is about to be sent, so the routing layer never learns
that the link is unavailable and no message is rerouted around it. The attack applies on top of the
ISL-only routing constraint rather than replacing it. Because the strategy takes directed pairs,
disabling a link requires supplying both orderings.

Command line
------------
--delivery            best_effort (default) or store_and_forward
--attack-start-time   default 0.0, active from the start of the run
--attack-probability  default 1.0, every message crossing the link is discarded
--target-link         default "26,27"
"""

import os
import sys
import pickle
import random
import argparse
from typing import Optional

from dsns.logging import BandwidthLoggingActor, LTPTransmissionLoggingActor, PreprocessedLoggingActor
from dsns.message_actors import BestEffortRoutingDataProvider, LookaheadRoutingDataProvider, MessageRoutingActor
from dsns.simulation import Simulation, LoggingActor
from dsns.message import LinkDownAttackStrategy
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

# Busiest inter-satellite link, identified from the baseline bandwidth records. Both satellites
# carry the full degree of four: this is the most heavily loaded link, not a structurally weak one.
DEFAULT_TARGET_LINK = (26, 27)

# The two comparison targets described in the module docstring. Not used by default; both can be
# selected explicitly with --target-link.
LOW_TRAFFIC_TARGET_LINK = (35, 34)     # 1 MB over a single active interval
ZERO_TRAFFIC_TARGET_LINK = (70, 71)    # never traversed by any message (negative control)

RESULTS_DIR_DEFAULT = os.path.join(_REPO_ROOT, "results", "link_down")


def build_link_down_attack(target_link: tuple[int, int], start_time: float, probability: float, seed: Optional[int] = 0):
    """Builds the attack strategy, supplying both orderings of the target pair.

    DSNS treats links as directed, so passing only (source, destination) would deny the link in one
    direction and leave the other working.
    """
    a, b = target_link
    links = {(a, b), (b, a)}
    return LinkDownAttackStrategy(
        links=links,
        start_time=start_time,
        probability=probability,
        seed=seed,
        message_filter=None,  # acts on all traffic crossing the link, no filtering
    )


def main(
        delivery: str = "best_effort",
        traffic_scale: int = 10,
        loss: Optional[float] = 0.005,
        walker_scale: int = 1,
        target_link: tuple[int, int] = DEFAULT_TARGET_LINK,
        attack_start_time: float = 0.0,
        attack_probability: float = 1.0,
        verbose: bool = False,
        results_dir: str = RESULTS_DIR_DEFAULT,
):
    if delivery not in ("best_effort", "store_and_forward"):
        raise ValueError(f"Unsupported delivery mode: {delivery} (expected best_effort or store_and_forward)")

    # --- Constellation: Iridium NEXT with the eight reconstructed ground stations ---------------
    constellation = IridiumReconstructedMultiConstellation(iridium_kwargs=dict(
        num_planes=6,
        sats_per_plane=11 * walker_scale,
    ))
    constellation.update(0.0)  # populates .isls/.ills/.satellites before the override is built

    ground_ids = set(constellation.ground_constellation.satellites.ids)
    ground_ids_list = list(ground_ids)

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

    # --- Ground-to-ground traffic, as in the baseline: ground stations only ---------------------
    message_size = 1e6  # 1 MB, in bytes
    message_rate = 10.0

    message_config = []
    for i in range(traffic_scale):
        random.seed(i)
        source, destination = random.sample(ground_ids_list, 2)
        message_config.append((f"Traffic-{i}", source, destination, message_size, message_rate))

    traffic_actor = MultiPointToPointTrafficActor(
        message_config=message_config,
        update_interval=60,
        reliable_messages=False,
        cutoff=6000,
    )
    actors.append(traffic_actor)

    # --- Random loss model, independent of the attack and identical to the baseline -------------
    loss_config = CustomScenarioLossConfig(
        seed=0,
        default_loss_probability=loss,
        max_frame_size=64 * 1024,
    ) if loss else None

    # --- The ground-to-ground, ISL-only routing constraint --------------------------------------
    next_hop_override = build_ground_to_ground_override(constellation, ground_ids)

    # --- The attack itself ----------------------------------------------------------------------
    attack_strategy = build_link_down_attack(
        target_link=target_link,
        start_time=attack_start_time,
        probability=attack_probability,
    )

    # --- Routing and delivery, combining the ISL-only constraint with the attack ----------------
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

    # --- Logging: the same three-file scheme used by every run ----------------------------------
    preprocessed_logging_actor = PreprocessedLoggingActor(log_other=False)
    bw_logging_actor = BandwidthLoggingActor()
    ltp_logging_actor = LTPTransmissionLoggingActor()
    logging_actors = [preprocessed_logging_actor, bw_logging_actor, ltp_logging_actor]
    if verbose:
        logging_actors.append(LoggingActor(verbose=True))

    # --- Running the simulation -----------------------------------------------------------------
    simulation = Simulation(
        constellation,
        actors=actors,
        logging_actors=logging_actors,
        data_providers=data_providers,
        timestep=15.0,
    )

    simulation.initialize(time=0)
    simulation.run(6000 * 4, progress=True)

    # --- Output: filename prefix carrying every attack parameter --------------------------------
    os.makedirs(results_dir, exist_ok=True)

    direct_messages = preprocessed_logging_actor.direct_messages
    broadcast_messages = preprocessed_logging_actor.broadcast_messages
    other_events = preprocessed_logging_actor.other_events

    default_bandwidth = transmission_actor._default_bandwidth
    period = 1.0

    a, b = target_link
    prefix = (
        f"iridium-g2g-linkdown-{delivery}-{loss}-point_to_point-{traffic_scale}-{walker_scale}"
        f"-target{a}_{b}-start{attack_start_time}-prob{attack_probability}"
    )

    with open(os.path.join(results_dir, f"{prefix}.pickle"), "wb") as f:
        pickle.dump((direct_messages, broadcast_messages, other_events), f)

    with open(os.path.join(results_dir, f"{prefix}-bw-{period}.pickle"), "wb") as f:
        pickle.dump(bw_logging_actor.aggregate(period=period, default_bandwidth=default_bandwidth), f)

    with open(os.path.join(results_dir, f"{prefix}-ltp-{period}.pickle"), "wb") as f:
        pickle.dump(ltp_logging_actor.aggregate(period=period), f)

    print(f"[OK] {prefix} -> {results_dir}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LinkDown attack on one Iridium NEXT inter-satellite link, on the "
                    "ground-to-ground, ISL-only model with eight reconstructed ground stations."
    )
    parser.add_argument("--delivery", type=str, choices=["best_effort", "store_and_forward"], default="best_effort")
    parser.add_argument("--traffic-scale", type=int, default=10)
    parser.add_argument("--loss", type=float, default=0.005,
                        help="Per-frame random loss probability, independent of the attack. 0.005 = 0.5%%.")
    parser.add_argument("--walker-scale", type=int, default=1)
    parser.add_argument("--target-link", type=str, default=f"{DEFAULT_TARGET_LINK[0]},{DEFAULT_TARGET_LINK[1]}",
                        help="Pair of satellite IDs to attack, for example '26,27' (default: the busiest "
                             "link). Also available: '35,34' (lightly loaded), '70,71' (never traversed).")
    parser.add_argument("--attack-start-time", type=float, default=0.0,
                        help="Simulated time in seconds at which the attack becomes active (default 0.0).")
    parser.add_argument("--attack-probability", type=float, default=1.0,
                        help="Drop probability for each message crossing the attacked link while the "
                             "attack is active (default 1.0, the link is denied completely).")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR_DEFAULT)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    a_str, b_str = args.target_link.split(",")
    target_link = (int(a_str.strip()), int(b_str.strip()))

    main(
        delivery=args.delivery,
        traffic_scale=args.traffic_scale,
        loss=args.loss if args.loss else None,
        walker_scale=args.walker_scale,
        target_link=target_link,
        attack_start_time=args.attack_start_time,
        attack_probability=args.attack_probability,
        verbose=args.verbose,
        results_dir=args.results_dir,
    )
