"""Paired baseline for the bandwidth exhaustion scenario.

Ordinary traffic with no adversary present, on the extended topology: the eight reconstructed
ground stations, the twenty-four synthetic user terminals and the sixty-six satellites. Two
point-to-point flows run from Fucino and from Tempe to Svalbard, on the message rate, size and
cutoff used throughout.

The terminals are present but silent, so they are part of both this run and the attacked one and
any difference between the two comes from the traffic they generate rather than from the extra
nodes themselves.

Ground station IDs are unchanged from the model without terminals, since the eight real sites are
assigned first: only the terminals and the satellites move further along the ID space. The ISL-only
override is reused from the baseline rather than duplicated, and the ground_ids passed to it remain
the eight real stations alone, so the terminals never act as endpoints for legitimate traffic.

Run once per delivery mode, each as a separate Python process: PreprocessedLoggingActor keeps state
across calls to main() within a single process, so two runs in one process would contaminate each
other.
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

FUCINO = 5
TEMPE = 0
SVALBARD = 3

RESULTS_DIR_DEFAULT = os.path.join(_REPO_ROOT, "results", "bandwidth_flood", "normal_traffic")


def main(
        delivery: str = "best_effort",
        loss: Optional[float] = 0.0,          # no channel loss in this scenario, by design
        message_rate: float = 10.0,           # invariato: coerente con latency_correction.py
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

    # The eight real ground stations alone. The synthetic terminals take no part in legitimate
    # traffic: they are the source of the attack, not correspondents of it.
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

    # Two simultaneous flows into Svalbard, from Fucino and from Tempe.
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

    prefix = f"iridium-ut-scenario-{delivery}-{loss}-point_to_point-fucino_tempe_to_svalbard-{walker_scale}"

    with open(os.path.join(results_dir, f"{prefix}.pickle"), "wb") as f:
        pickle.dump((direct_messages, broadcast_messages, other_events), f)

    with open(os.path.join(results_dir, f"{prefix}-bw-{period}.pickle"), "wb") as f:
        pickle.dump(bw_logging_actor.aggregate(period=period, default_bandwidth=default_bandwidth), f)

    with open(os.path.join(results_dir, f"{prefix}-ltp-{period}.pickle"), "wb") as f:
        pickle.dump(ltp_logging_actor.aggregate(period=period), f)

    # Statistics recomputed by reading back the pickle just written, so the file on disk is the
    # single source of truth.
    with open(os.path.join(results_dir, f"{prefix}.pickle"), "rb") as f:
        saved_direct_messages, _, _ = pickle.load(f)
    msgs = list(saved_direct_messages.values())
    fucino_msgs = [m for m in msgs if m.source == FUCINO]
    tempe_msgs = [m for m in msgs if m.source == TEMPE]
    for label, sub in (("Fucino->Svalbard", fucino_msgs), ("Tempe->Svalbard", tempe_msgs)):
        n_sub = len(sub)
        delivered = sum(1 for m in sub if m.delivered)
        dropped = sum(1 for m in sub if m.dropped)
        limbo = n_sub - delivered - dropped
        print(f"  {label:20s} n={n_sub}  delivered={delivered} ({100*delivered/n_sub:.1f}%)  "
              f"dropped={dropped} ({100*dropped/n_sub:.1f}%)  limbo={limbo} ({100*limbo/n_sub:.1f}%)")

    print(f"\n[OK] {prefix} -> {results_dir}")


if __name__ == "__main__":
    main()
