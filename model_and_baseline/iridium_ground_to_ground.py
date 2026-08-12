"""Ground-to-ground model with ISL-only intermediate hops, and the baseline traffic run.

Constrains message paths to the structure real user traffic actually takes: source and destination
are always ground stations, and every intermediate hop is a satellite reached over an
inter-satellite link, with no bounce back down to a ground station mid-path.

    source = ground station -> sat -> sat -> ... -> sat -> ground station = destination

Why the constraint exists
-------------------------
DSNS draws no distinction between the roles a node can play: routing treats every node in the graph
alike, so any satellite or ground station can equally be assigned as a message source or
destination. Iridium NEXT does not work that way. A satellite is always an intermediate relay,
carrying traffic that originates and terminates in the ground or user segment; it is never itself
the party sending or receiving application data.

This script adds that topological constraint without introducing any simulation of distributed
routing. Next-hop computation still assumes complete, instantaneous knowledge of the whole
topology, the same idealisation the DSNS authors acknowledge for their own default routing. What
changes is which nodes are eligible at each hop, not how the decision is made.

How the constraint works, without modifying DSNS
------------------------------------------------
It uses get_next_hop_override, a hook already exposed by the simulator and accepted by the
constructors of both BestEffortRoutingDataProvider and LookaheadRoutingDataProvider, which forward
it to RoutingDataProvider:

    def get_next_hop(self, source, destination, message=None):
        if self.get_next_hop_override:
            next_hop = self.get_next_hop_override(message, source, destination)
            if next_hop is not None:
                return next_hop
        return self._get_next_hop(source, destination)   # default behaviour, unchanged

Returning None therefore leaves DSNS to behave exactly as it would without the override, and the
routing provider's own internal graph is never touched. The override keeps a separate graph
(networkx) weighted with the same delay values DSNS uses internally, so the notion of a shortest
path stays physically consistent between the two: only the set of admissible nodes differs.

At every hop the override does the following:

  1. From a ground station, return None. Ground stations have no direct links to one another, so
     the only reachable next hop is already a satellite and default routing can proceed unchanged.
  2. From a satellite, check which satellites currently see the destination ground station.
       a. If the current satellite is among them, return the destination directly and land.
       b. Otherwise compute the shortest path over inter-satellite links alone toward the nearest
          satellite that does see the destination, and return the first hop of that path. If no
          satellite sees the destination at that instant, or no ISL-only path reaches one, return
          None and fall back to the simulator's own behaviour.

Ordering guarantees
-------------------
Simulation.step() calls mobility.update() before running any actor, so the ISLs and ILLs read
inside the override are always those of the current instant. The constellation object held here is
the same live object the simulation updates, not a copy, so there is no window in which a stale
topology could be read. DSNS is single-threaded and event-driven, so no concurrency arises within a
run, and parallel runs are separate OS processes with separate memory.

Ground segment
--------------
In place of the 256 synthetic, uniformly distributed positions DSNS generates by default, this
script uses the eight reconstructed Iridium NEXT sites. With eight ground stations rather than 256,
ground-to-ground traffic is necessarily concentrated on a small number of possible pairs, which is
expected.
"""

import os
import sys
import pickle
import random
import argparse
from typing import Optional

import networkx as nx

from dsns.logging import BandwidthLoggingActor, LTPTransmissionLoggingActor, PreprocessedLoggingActor
from dsns.message_actors import BestEffortRoutingDataProvider, LookaheadRoutingDataProvider, MessageRoutingActor
from dsns.simulation import Simulation, LoggingActor
from dsns.message import Link, LossConfig
from dsns.transmission import MessageLocationTracker, LinkTransmissionActor
from dsns.traffic_sim import MultiPointToPointTrafficActor, RandomTrafficActor, NormalSampler, UniformSampler

# The repository is a set of flat scripts rather than a package, so the repository root is put on
# the path explicitly to import the shared modules (ground_stations_reconstructed,
# presets_reconstructed), which live one level above this file.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "model_and_baseline"),
    os.path.dirname(os.path.abspath(__file__)),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from presets_reconstructed import IridiumReconstructedMultiConstellation  # noqa: E402


RESULTS_DIR_DEFAULT = os.path.join(_REPO_ROOT, "results", "baseline")


class CustomScenarioLossConfig(LossConfig):
    """Per-frame random loss model, applied independently of any attack."""

    def __init__(self, seed: float = 0, default_loss_probability: float = 0, max_frame_size: int = 64 * 1024):
        super().__init__(seed, default_loss_probability)
        self.max_frame_size = max_frame_size

    def is_message_lost(self, source: int, destination: int, size: int) -> bool:
        link = Link(source=source, destination=destination)
        rng = self._get_rng_for_link(link)
        loss_probability = self._get_loss_probability_for_link(link)
        num_frames = (size + self.max_frame_size - 1) // self.max_frame_size
        for _ in range(num_frames):
            if rng.random() < loss_probability:
                return True
        return False


def build_ground_to_ground_override(constellation, ground_ids: set):
    """Builds the function passed to get_next_hop_override, as described in the module docstring.

    The ISL-only graph is built once from the static edge list, since the inter-satellite topology
    does not change, but the edge weights are refreshed on every call from constellation.get_delay(),
    matching how the simulator recomputes the weights of its own graph at each timestep.
    """
    isl_graph = nx.Graph()
    for s1, s2 in constellation.isls:
        isl_graph.add_edge(s1, s2, weight=constellation.get_delay(s1, s2))

    def override(message, source, destination):
        if source in ground_ids:
            # From a ground station the only reachable next hop is already a satellite: DSNS has
            # no direct ground-to-ground links here, so default routing can proceed unchanged.
            return None

        # Refresh the edge weights: distances change over time even when the ISL topology is
        # fixed, following the same principle as the simulator's own periodic recomputation.
        for s1, s2 in isl_graph.edges():
            isl_graph[s1][s2]["weight"] = constellation.get_delay(s1, s2)

        # Satellites that can currently see the destination ground station.
        visible_sats = {
            (a if b == destination else b)
            for a, b in constellation.ills
            if destination in (a, b)
        }

        if source in visible_sats:
            return destination  # final delivery: this satellite already sees the destination

        if not visible_sats or source not in isl_graph:
            return None  # no satellite sees the destination now, or this one is isolated

        lengths, paths = nx.single_source_dijkstra(isl_graph, source, weight="weight")

        best_hop, best_dist = None, None
        for target_sat in visible_sats:
            if target_sat not in lengths:
                continue
            dist = lengths[target_sat]
            if best_dist is None or dist < best_dist:
                best_dist = dist
                path = paths[target_sat]
                best_hop = path[1] if len(path) > 1 else target_sat

        return best_hop  # None if no ISL-only path reaches any satellite currently in view

    return override


def main(
        delivery: str = "best_effort",
        traffic: str = "point_to_point",
        traffic_scale: int = 10,
        loss: Optional[float] = 0.005,
        walker_scale: int = 1,
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
    sat_ids = list(constellation.iridium_constellation.satellites.ids)
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

    # --- Traffic generation: between ground stations only, never a satellite as an endpoint -----
    if traffic in ("point_to_point", "point_to_point_eos"):
        message_size = 1e6 if traffic == "point_to_point" else 10e6
        message_rate = 10.0 if traffic == "point_to_point" else 11.5

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
    elif traffic == "random":
        traffic_actor = RandomTrafficActor(
            satellites=ground_ids_list,
            message_interval=1.0 / traffic_scale,
            message_size=NormalSampler(mean=1e6, std=1e6 // 4),
            message_source=UniformSampler(min=0, max=len(ground_ids_list) - 1),
            message_destination=UniformSampler(min=0, max=len(ground_ids_list) - 1),
            update_interval=60,
            reliable_messages=False,
        )
    else:
        raise ValueError(f"Unknown traffic type: {traffic}")

    actors.append(traffic_actor)

    # --- Loss model -----------------------------------------------------------------------------
    loss_config = CustomScenarioLossConfig(
        seed=0,
        default_loss_probability=loss,
        max_frame_size=64 * 1024,
    ) if loss else None

    # --- The ground-to-ground, ISL-only routing constraint --------------------------------------
    next_hop_override = build_ground_to_ground_override(constellation, ground_ids)

    # --- Routing e delivery -----------------------------------------------------------------
    if delivery == "best_effort":
        routing_data_provider = BestEffortRoutingDataProvider(get_next_hop_override=next_hop_override)
        routing_actor = MessageRoutingActor(
            routing_data_provider,
            store_and_forward=False,
            model_bandwidth=True,
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
            loss_config=loss_config,
        )

    actors.append(routing_actor)
    data_providers.append(routing_data_provider)

    # --- Logging (identico a iridium_baseline.py) ------------------------------------------------
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

    # --- Saving the results -----------------------------------------------------------------------
    os.makedirs(results_dir, exist_ok=True)

    direct_messages = preprocessed_logging_actor.direct_messages
    broadcast_messages = preprocessed_logging_actor.broadcast_messages
    other_events = preprocessed_logging_actor.other_events

    default_bandwidth = transmission_actor._default_bandwidth
    period = 1.0

    # The "g2g" prefix marks this scenario: ground-to-ground endpoints, ISL-only intermediate
    # hops, reconstructed ground stations.
    prefix = f"iridium-g2g-{delivery}-{loss}-{traffic}-{traffic_scale}-{walker_scale}"

    with open(os.path.join(results_dir, f"{prefix}.pickle"), "wb") as f:
        pickle.dump((direct_messages, broadcast_messages, other_events), f)

    with open(os.path.join(results_dir, f"{prefix}-bw-{period}.pickle"), "wb") as f:
        pickle.dump(bw_logging_actor.aggregate(period=period, default_bandwidth=default_bandwidth), f)

    with open(os.path.join(results_dir, f"{prefix}-ltp-{period}.pickle"), "wb") as f:
        pickle.dump(ltp_logging_actor.aggregate(period=period), f)

    print(f"[OK] {prefix} -> {results_dir}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Iridium NEXT ground-to-ground traffic: endpoints are always reconstructed "
                    "ground stations and every intermediate hop is an inter-satellite link."
    )
    parser.add_argument("--delivery", type=str, choices=["best_effort", "store_and_forward"], default="best_effort")
    parser.add_argument("--traffic", type=str, choices=["point_to_point", "point_to_point_eos", "random"], default="point_to_point")
    parser.add_argument("--traffic-scale", type=int, default=10)
    parser.add_argument("--loss", type=float, default=0.005)
    parser.add_argument("--walker-scale", type=int, default=1)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR_DEFAULT)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(
        delivery=args.delivery,
        traffic=args.traffic,
        traffic_scale=args.traffic_scale,
        loss=args.loss if args.loss else None,
        walker_scale=args.walker_scale,
        verbose=args.verbose,
        results_dir=args.results_dir,
    )
