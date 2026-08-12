"""Three-dimensional rendering of the path a single message follows.

Reads a pickle of events produced by message_path_trace.py and reconstructs the highlighted paths
from the data rather than from a command line argument: MessageSentEvent entries whose message.data
matches the chosen label, for instance "Traffic-0-0", are filtered out, ordered by time, and their
(source, destination) pairs read in sequence. That ordered sequence of hops is the trajectory the
message actually followed.

TwoColorEarthVisualizer, run_viewer_with_save_dir, ASSETS_DIR and SCREENSHOT_SUPERSAMPLE are
imported from visualize_iridium_earth rather than duplicated. Drawing satellites and ground stations
in separate colours matters here: it makes immediately visible which nodes of a path are ground
stations, at the two ends, and which are satellites carrying it in between. Highlight groups are
supported by that class exactly as they are by the simulator's own highlighting visualizer, so the
substitution required no change to the path extraction below.

The constellation is IridiumReconstructedMultiConstellation with the eight reconstructed ground
stations, built with the same arguments as the baseline. The default pickle is the one produced by
message_path_trace.py, under results/message_paths with the "iridium-g2g" prefix, and renders are
written to results/renders.

final_outcome() reports a message left unresolved as such, and names both possible causes: the loss
model, and the timestep quantization behaviour. The traces this script reads are always produced
under best-effort delivery, which is the mode the second of those affects.

Usage
-----
  python3 visualize_message_path.py --test --output-file path.png
  python3 visualize_message_path.py --message-data Traffic-0-0 --message-data Traffic-1-0
  python3 visualize_message_path.py --pickle-path /some/other/file-messagepath.pickle
"""

import argparse
import os
import pickle
import re
import sys
import time

import numpy as np
import pyrender

# The two-colour 3D rendering infrastructure is imported from visualize_iridium_earth rather than
# duplicated here.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "model_and_baseline"),
    os.path.dirname(os.path.abspath(__file__)),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from visualize_iridium_earth import (  # noqa: E402
    TwoColorEarthVisualizer,
    run_viewer_with_save_dir,
    ASSETS_DIR,
    SCREENSHOT_SUPERSAMPLE,
)

# The repository is a set of flat scripts rather than a package, so the repository root is put on
# the path explicitly to import presets_reconstructed.
from presets_reconstructed import IridiumReconstructedMultiConstellation  # noqa: E402

from dsns.message import MessageSentEvent, MessageDeliveredEvent, MessageDroppedEvent  # noqa: E402

OUT_DIR = os.path.join(_REPO_ROOT, "results", "renders")
MESSAGE_PATHS_DIR = os.path.join(_REPO_ROOT, "results", "message_paths")
DEFAULT_PICKLE_PATH = (
    f"{MESSAGE_PATHS_DIR}/iridium-g2g-best_effort-None-point_to_point-3-1-messagepath.pickle"
)

# Clearly distinguishable colours, cycled if there are more messages to trace than colours.
PATH_COLOR_PALETTE: list[tuple[float, float, float]] = [
    (1.0, 1.0, 0.0),   # giallo
    (0.0, 1.0, 1.0),   # ciano
    (1.0, 0.0, 1.0),   # magenta
    (1.0, 0.5, 0.0),   # arancione
    (0.5, 1.0, 0.0),   # verde lime
    (1.0, 1.0, 1.0),   # bianco
]


def load_events(pickle_path: str) -> list:
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def discover_default_message_labels(events: list) -> list[str]:
    """Without an explicit --message-data, traces message 0 of each stream found in the pickle.

    Streams are taken in increasing order and identified by the Traffic-{stream}-0 label.
    """
    label_re = re.compile(r"^Traffic-(\d+)-(\d+)$")
    streams_seen = set()
    for e in events:
        message = getattr(e, "message", None)
        if message is None:
            continue
        m = label_re.match(message.data)
        if m:
            streams_seen.add(int(m.group(1)))
    return [f"Traffic-{stream}-0" for stream in sorted(streams_seen)]


def extract_path(events: list, message_label: str) -> list[tuple[int, int]]:
    """Orders the send events for a given message label by time and derives its hop sequence.

    By construction the first hop leaves a ground station and the last arrives at one, with every
    intermediate hop running satellite to satellite under the ISL-only constraint.
    """
    sent_events = [
        e for e in events
        if isinstance(e, MessageSentEvent) and getattr(e, "message", None) is not None
        and e.message.data == message_label
    ]
    sent_events.sort(key=lambda e: e.time)
    return [(e.source, e.destination) for e in sent_events]


def final_outcome(events: list, message_label: str) -> str:
    """Reports a message as delivered, dropped with its reason, or unresolved.

    Two distinct behaviours leave a message unresolved and are indistinguishable from this trace
    alone. The loss model raises a MessageLostEvent, which the generic LoggingActor used here never
    intercepts because it filters only the create, send, receive, deliver and drop events. Separately,
    timestep quantization can advance the clock just past an event, so a link going down within the
    resulting gap is never recorded. The second affects best-effort delivery only, which is the mode
    these traces always use.
    """
    for e in events:
        message = getattr(e, "message", None)
        if message is None or message.data != message_label:
            continue
        if isinstance(e, MessageDeliveredEvent):
            return f"CONSEGNATO (t={e.time:.2f}s)"
        if isinstance(e, MessageDroppedEvent):
            return f"SCARTATO (motivo={e.reason}, t={e.time:.2f}s)"
    return (
        "UNRESOLVED (neither delivered nor dropped: either the loss model, whose MessageLostEvent "
        "this trace does not record, or timestep quantization, which affects best-effort delivery "
        "only. The two are indistinguishable from this trace without also inspecting link events)"
    )


def print_path_summary(events: list, message_label: str, path: list[tuple[int, int]],
                        color: tuple[float, float, float]) -> None:
    if not path:
        print(f"WARNING: no hop found for '{message_label}'. Check that the label exists in the pickle.")
        return
    hops_str = " -> ".join(str(path[0][0]) for _ in [0]) + "".join(f" -> {d}" for _, d in path)
    outcome = final_outcome(events, message_label)
    print(f"{message_label} (RGB {color}): {len(path)} hops, path: {hops_str} - {outcome}")


def build_message_path_visualizer(viewport_size, time_scale, space_scale, white_bg, highlight_groups,
                                   walker_scale: int = 1, ground_color: tuple[float, float, float] = (0.0, 1.0, 0.0)):
    """Builds the visualizer with the eight reconstructed ground stations.

    Used in place of the 256 synthetic defaults, with the same constellation arguments as the
    baseline, so that the rendered constellation
    matches exactly the one the traced simulation was generated on.

    Satellites are drawn red and ground stations green and larger, so that along a highlighted path
    it is immediately clear which nodes are the endpoints and which are relays.
    """
    earth_materials = (
        f"{ASSETS_DIR}/2k_earth_daymap.jpg",
        f"{ASSETS_DIR}/2k_earth_normal_map.jpg",
        f"{ASSETS_DIR}/2k_earth_metallic_roughness_map.jpg",
    )
    bg_color = (1.0, 1.0, 1.0) if white_bg else (0.0, 0.0, 0.0)

    constellation = IridiumReconstructedMultiConstellation(iridium_kwargs=dict(
        num_planes=6,
        sats_per_plane=11 * walker_scale,
    ))
    return TwoColorEarthVisualizer(
        constellation,
        time_scale=time_scale,
        space_scale=space_scale,
        earth_materials=earth_materials,
        viewport_size=viewport_size,
        bg_color=bg_color,
        ground_color=ground_color,
        highlight_groups=highlight_groups,
    )


def run_interactive(viewport_size, time_scale, space_scale, white_bg, highlight_groups, walker_scale=1,
                     ground_color=(0.0, 1.0, 0.0)):
    visualizer = build_message_path_visualizer(viewport_size, time_scale, space_scale, white_bg, highlight_groups, walker_scale, ground_color)
    visualizer.update_simulation(0.0)
    visualizer.run_simulation()
    rebuild_fn = lambda: build_message_path_visualizer(
        viewport_size, time_scale, space_scale, white_bg, highlight_groups, walker_scale, ground_color
    )
    run_viewer_with_save_dir(visualizer, OUT_DIR, prefix="message_path_iridium_g2g", rebuild_fn=rebuild_fn)

    while visualizer.viewer is None or visualizer.viewer.is_active:
        time.sleep(0.1)


def run_offscreen_test(viewport_size, time_scale, space_scale, white_bg, output_file, highlight_groups, walker_scale=1,
                        ground_color=(0.0, 1.0, 0.0)):
    os.makedirs(OUT_DIR, exist_ok=True)

    visualizer = build_message_path_visualizer(viewport_size, time_scale, space_scale, white_bg, highlight_groups, walker_scale, ground_color)
    visualizer.update_simulation(0.0)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=viewport_size[0] / viewport_size[1])
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 20.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    visualizer.scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    visualizer.scene.add(light, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(
        viewport_width=viewport_size[0] * SCREENSHOT_SUPERSAMPLE,
        viewport_height=viewport_size[1] * SCREENSHOT_SUPERSAMPLE,
    )
    color, depth = renderer.render(visualizer.scene)
    renderer.delete()

    out_path = os.path.join(OUT_DIR, output_file)
    import PIL.Image
    PIL.Image.fromarray(color).save(out_path)
    print(f"[OK] Offscreen screenshot saved to {out_path} ({color.shape[1]}x{color.shape[0]})")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Renders in 3D the real hop-by-hop path of one or more messages, read from a "
                    "pickle produced by message_path_trace.py."
    )
    parser.add_argument("--pickle-path", type=str, default=None,
                         help="Path to the *-messagepath.pickle file to read events from. "
                              "Defaults to the standard baseline trace.")
    parser.add_argument("--message-data", type=str, action="append", default=[], dest="message_labels",
                         help="Label of the message to trace (for example 'Traffic-0-0'), repeatable "
                              "for several paths. Defaults to the first message of each stream "
                              "found in the pickle, each in a different colour.")
    parser.add_argument("--walker-scale", type=int, default=1,
                         help="Must match the --walker-scale used to generate the pickle, "
                              "otherwise the rendered constellation will not match the "
                              "simulated one.")
    parser.add_argument("--time-scale", type=float, default=100.0)
    parser.add_argument("--space-scale", type=float, default=1e-6)
    parser.add_argument("--viewport-size", type=int, nargs=2, default=(800, 600))
    parser.add_argument("-w", "--white-bg", action="store_true")
    parser.add_argument("--test", action="store_true",
                         help="Do not open the interactive window: render offscreen and save a PNG screenshot")
    parser.add_argument("--output-file", type=str, default="message_path_test.png")
    parser.add_argument("--ground-color", type=float, nargs=3, default=(0.0, 1.0, 0.0),
                         metavar=("R", "G", "B"), help="RGB colour (0-1) for the ground stations, green by default.")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    viewport_size = tuple(args.viewport_size)
    pickle_path = args.pickle_path or DEFAULT_PICKLE_PATH

    events = load_events(pickle_path)
    message_labels = args.message_labels or discover_default_message_labels(events)
    if not message_labels:
        raise ValueError(f"No message found in pickle {pickle_path!r}")

    highlight_groups = []
    for i, label in enumerate(message_labels):
        color = PATH_COLOR_PALETTE[i % len(PATH_COLOR_PALETTE)]
        path = extract_path(events, label)
        print_path_summary(events, label, path, color)
        if path:
            highlight_groups.append((path, color))

    ground_color = tuple(args.ground_color)

    if args.test:
        run_offscreen_test(viewport_size, args.time_scale, args.space_scale, args.white_bg,
                            args.output_file, highlight_groups, args.walker_scale, ground_color)
    else:
        run_interactive(viewport_size, args.time_scale, args.space_scale, args.white_bg,
                         highlight_groups, args.walker_scale, ground_color)
