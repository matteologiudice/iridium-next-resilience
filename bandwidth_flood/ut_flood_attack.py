"""Distributed bandwidth exhaustion from compromised user terminals.

Twenty-four synthetic user terminals, spread across three geographic regions, transmit toward
Svalbard for ninety seconds, from 990 to 1,080 seconds of simulated time. Each emits one megabyte
every two seconds, which is 4 Mbps, so the eight terminals of a region converge on the satellite
serving that region for an aggregate of roughly 32 Mbps against the 25 Mbps a link provides.

Nothing in the routing is forced. Each terminal simply addresses Svalbard and the routing layer
selects its path exactly as it does for legitimate traffic, which is why the geographic
distribution of the terminals, rather than any explicit targeting, determines which satellites the
load reaches.

Choice of window
----------------
The window falls inside an interval in which Svalbard reaches its maximum number of simultaneous
satellite links, so the congestion reaches as much of the constellation's traffic as the ground
segment allows. It was also checked against the paired baseline and contains none of the messages
that the baseline itself leaves unresolved elsewhere in the run, so nothing in the measured effect
predates the attack. Routing stability across the window was verified separately: all
twenty-four terminals keep at least one satellite in common throughout the interval.

The traffic-generating actor is SimpleFloodActor, imported from flood_actor rather than
duplicated.
"""

import os
import sys
import pickle
from typing import Optional

from dsns.logging import BandwidthLoggingActor, LTPTransmissionLoggingActor, PreprocessedLoggingActor
from dsns.message_actors import BestEffortRoutingDataProvider, LookaheadRoutingDataProvider, MessageRoutingActor
from dsns.simulation import Simulation, LoggingActor
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

from presets_base_station import IridiumUserTerminalMultiConstellation  # noqa: E402
from iridium_ground_to_ground import CustomScenarioLossConfig, build_ground_to_ground_override  # noqa: E402
from flood_actor import SimpleFloodActor  # noqa: E402

FUCINO = 5
TEMPE = 0
SVALBARD = 3

FLOOD_START = 990.0
FLOOD_END = 1080.0
FLOOD_SIZE = 1_000_000
FLOOD_PERIOD = 2.0   # 4 Mbps per compromised terminal. With eight terminals converging on the
                     # satellite serving their region, roughly 32 Mbps against a 25 Mbps link.

RESULTS_DIR_DEFAULT = os.path.join(_REPO_ROOT, "results", "bandwidth_flood", "under_attack")


def main(
        delivery: str = "best_effort",
        loss: Optional[float] = 0.0,
        message_rate: float = 10.0,
        walker_scale: int = 1,
        verbose: bool = False,
        results_dir: str = RESULTS_DIR_DEFAULT,
):
    if delivery not in ("best_effort", "store_and_forward"):
        raise ValueError(f"Unsupported delivery mode: {delivery} (expected best_effort or store_and_forward)")

    constellation = IridiumUserTerminalMultiConstellation(iridium_kwargs=dict(
        num_planes=6,
        sats_per_plane=11 * walker_scale,
    ))
    constellation.update(0.0)

    gs_ids = set(constellation.ground_constellation.satellites.ids)
    ut_ids = sorted(constellation.ut_constellation.satellites.ids)
    ground_ids = gs_ids | set(ut_ids)

    compromised_uts = ut_ids  # all twenty-four terminals

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
    actors.append(traffic_actor)

    flood_links = [(ut, SVALBARD) for ut in compromised_uts]
    flood_actor = SimpleFloodActor(
        links=flood_links,
        period=FLOOD_PERIOD,
        size=FLOOD_SIZE,
        start_time=FLOOD_START,
        end_time=FLOOD_END,
    )
    actors.append(flood_actor)

    loss_config = CustomScenarioLossConfig(seed=0, default_loss_probability=loss, max_frame_size=64 * 1024) if loss else None

    next_hop_override = build_ground_to_ground_override(constellation, ground_ids)

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

    prefix = f"iridium-ut-flood-{delivery}-{loss}-point_to_point-fucino_tempe_to_svalbard-{walker_scale}"

    with open(os.path.join(results_dir, f"{prefix}.pickle"), "wb") as f:
        pickle.dump((direct_messages, broadcast_messages, other_events), f)

    with open(os.path.join(results_dir, f"{prefix}-bw-{period}.pickle"), "wb") as f:
        pickle.dump(bw_logging_actor.aggregate(period=period, default_bandwidth=default_bandwidth), f)

    with open(os.path.join(results_dir, f"{prefix}-ltp-{period}.pickle"), "wb") as f:
        pickle.dump(ltp_logging_actor.aggregate(period=period), f)

    with open(os.path.join(results_dir, f"{prefix}.pickle"), "rb") as f:
        saved_direct_messages, _, _ = pickle.load(f)
    msgs = list(saved_direct_messages.values())
    legit_msgs = [m for m in msgs if m.source in (FUCINO, TEMPE)]
    flood_msgs = [m for m in msgs if m.source in compromised_uts]

    print(f"\n=== UT bandwidth exhaustion (24 terminals, 8 per region, 4 Mbps each, window [990, 1080] s)  delivery={delivery} ===")
    for label, source in (("Fucino->Svalbard", FUCINO), ("Tempe->Svalbard", TEMPE)):
        sub = [m for m in legit_msgs if m.source == source]
        n_sub = len(sub)
        delivered = sum(1 for m in sub if m.delivered)
        dropped = sum(1 for m in sub if m.dropped)
        limbo = n_sub - delivered - dropped
        delivered_latencies = [(m.end_time - m.start_time) for m in sub if m.delivered]
        avg_latency = sum(delivered_latencies) / len(delivered_latencies) if delivered_latencies else float("nan")
        max_latency = max(delivered_latencies) if delivered_latencies else float("nan")
        print(f"  {label:20s} n={n_sub}  delivered={delivered} ({100*delivered/n_sub:.1f}%)  "
              f"dropped={dropped} ({100*dropped/n_sub:.1f}%)  limbo={limbo} ({100*limbo/n_sub:.1f}%)  "
              f"latenza media (grezza)={avg_latency:.2f}s  latenza max (grezza)={max_latency:.2f}s")

    flood_delivered = sum(1 for m in flood_msgs if m.delivered)
    print(f"  Flood messages generated: {len(flood_msgs)} (from {len(compromised_uts)} compromised terminals, "
          f"consegnati={flood_delivered})")
    print(f"\n[OK] {prefix} -> {results_dir}")


if __name__ == "__main__":
    main()
