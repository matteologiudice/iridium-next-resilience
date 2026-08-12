"""Three-dimensional rendering of message paths during the bandwidth exhaustion attack.

Shows legitimate traffic and flood traffic converging on the same satellites, over the attack
window from 990 to 1,080 seconds.

Which messages are drawn is discovered from the data rather than fixed by ID. For each legitimate
flow, one delivered example is selected; among the flood messages, the ones actually sharing a
satellite with a legitimate path are preferred, so the rendering shows real contention rather than
a terminal picked arbitrarily. Only satellites count for that intersection: Svalbard is the
destination of every message, legitimate and flood alike, so including ground nodes would make the
test trivially true.

Tubes drawn along a shared link are laterally offset, so paths crossing the same hop stay visible
instead of hiding one another. The render time is the mean of the first hops of the highlighted
paths rather than zero, so the constellation is drawn at the moment those hops actually occurred.

The visualizer class, path extraction and colour palette are imported from the modules that already
define them rather than duplicated. Flood messages are matched by message UID rather than by label,
since their textual label is not unique across terminals.
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import pyrender
import trimesh
import trimesh.creation

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "model_and_baseline"),
    os.path.dirname(os.path.abspath(__file__)),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from visualize_user_terminal_earth import (  # noqa: E402  reused, not duplicated
    ThreeColorEarthVisualizer,
    run_viewer_with_save_dir,
    ASSETS_DIR,
    SCREENSHOT_SUPERSAMPLE,
    IridiumUserTerminalMultiConstellation,
)
import visualize_message_path as vmp  # noqa: E402  reuses extract_path, load_events and the colour palette
from visualize_message_path_offset import _tube_transform_for_segment  # noqa: E402  same function, reused

from dsns.message import MessageDeliveredEvent, MessageSentEvent  # noqa: E402

FUCINO = 5
TEMPE = 0
SVALBARD = 3
FLOOD_START = 990.0
FLOOD_END = 1080.0

OUT_DIR = os.path.join(_REPO_ROOT, "results", "renders")
DEFAULT_PICKLE_PATH = os.path.join(
    _REPO_ROOT, "results", "bandwidth_flood", "message_paths",
    "ut-flood-messagepath-best_effort.pickle",
)

LEGIT_LABEL_RE = re.compile(r"^Traffic-([01])-(\d+)$")
FLOOD_LABEL_RE = re.compile(r"^Flooding message: (\d+) -> (\d+)$")

FLOOD_COLOR = (1.0, 0.0, 0.0)  # reserved for flood messages, never reused for legitimate traffic
LEGIT_STREAM_LABELS = {"0": "Fucino->Svalbard", "1": "Tempe->Svalbard"}


def is_delivered(events: list, message_label: str) -> bool:
    return any(
        isinstance(e, MessageDeliveredEvent) and getattr(e, "message", None) is not None
        and e.message.data == message_label
        for e in events
    )


def first_sent_time(events: list, message_label: str) -> float | None:
    times = [
        e.time for e in events
        if isinstance(e, MessageSentEvent) and getattr(e, "message", None) is not None
        and e.message.data == message_label
    ]
    return min(times) if times else None


def discover_legit_labels(events: list, n_per_stream: int = 1,
                           window: tuple[float, float] = (FLOOD_START, FLOOD_END)) -> list[str]:
    """One delivered example per legitimate flow, 0 for Fucino and 1 for Tempe.

    Chosen among messages whose first hop falls inside the attack window, so the visual comparison
    is temporally consistent with the congestion being shown.
    """
    by_stream: dict[str, list[tuple[str, float]]] = {"0": [], "1": []}
    for e in events:
        msg = getattr(e, "message", None)
        if msg is None:
            continue
        m = LEGIT_LABEL_RE.match(msg.data)
        if not m:
            continue
        stream = m.group(1)
        label = msg.data
        if any(label == l for l, _ in by_stream[stream]):
            continue
        t0 = first_sent_time(events, label)
        if t0 is None or not (window[0] <= t0 <= window[1]):
            continue
        if not is_delivered(events, label):
            continue
        by_stream[stream].append((label, t0))

    selected = []
    for stream in ("0", "1"):
        candidates = sorted(by_stream[stream], key=lambda lt: lt[1])
        selected.extend(label for label, _ in candidates[:n_per_stream])
    return selected


# Global ID ranges of the twenty-four synthetic user terminals. The ground constellation is added
# first, so the operator stations take IDs 0 to 7 and the terminals follow
# User terminals occupy IDs 8 to 31, in the order the three regions are generated: Europe and
# western Russia (8 to 15), the Pacific coast of the United States (16 to 23), and its central and
# eastern seaboard (24 to 31). The two American regions are treated as one range here, since the
# rendering distinguishes continents rather than coasts.
RUSSIA_EUROPE_UT_RANGE = range(8, 16)
USA_UT_RANGE = range(16, 32)

# First satellite ID: stations and terminals occupy IDs 0 to 31, so satellites start at 32. Used
# to exclude ground nodes from preferred_nodes in build_highlight_groups, since Svalbard
# is the destination of every message, flood and legitimate alike. Leaving it in the preferred set
# would make the intersection always non-empty, so the "shares a satellite with the legitimate
# path" filter would never actually select anything and would fall back to the first terminal in
# the range by UID order.
FIRST_SATELLITE_ID = 32


def discover_flood_examples(events: list,
                             region_specs: list[tuple[range, frozenset, float | None]] = (
                                 (RUSSIA_EUROPE_UT_RANGE, frozenset(), None),
                                 (USA_UT_RANGE, frozenset(), None),
                             )) -> list[tuple[str, int]]:
    """One delivered example for each range of node IDs given.

    Each entry of region_specs is (id_range, preferred_nodes, reference_time). When preferred_nodes
    is not empty, the search runs over every delivered flood message from any terminal in that
    range, not just the first, and picks the one whose path touches at least one of the preferred
    satellites, typically those carrying the legitimate traffic. The message shown then visibly
    converges with the legitimate one instead of merely coinciding by region. If nothing satisfies
    the intersection, the first available terminal is used as a fallback, with a warning.

    When reference_time is given, the candidate chosen among those intersecting is the one whose
    first hop is closest to that instant, rather than the first by UID, which is an arbitrary order.
    This matters for geometric correctness: satellites are drawn at a single instant, so a flood
    example far in time from the legitimate traffic would draw hops over links that do not exist at
    that moment.

    Selection is by message UID rather than by label. SimpleFloodActor reuses the same textual label
    for every message a given terminal generates during the attack window, unlike legitimate traffic,
    which labels each message uniquely. Filtering by label alone would therefore concatenate the hops
    of dozens of distinct messages into one meaningless path. Returns (label, uid).
    """
    by_ut_uids: dict[int, list[tuple[int, str]]] = {}
    seen_uids = set()
    for e in events:
        msg = getattr(e, "message", None)
        if msg is None or msg.uid in seen_uids:
            continue
        m = FLOOD_LABEL_RE.match(msg.data)
        if not m:
            continue
        seen_uids.add(msg.uid)
        ut = int(m.group(1))
        by_ut_uids.setdefault(ut, []).append((msg.uid, msg.data))

    # Index from UID to time-ordered hops, plus the set of delivered UIDs, all built in a single
    # pass: selecting by temporal proximity has to examine every candidate in a region, of which
    # there are hundreds, rather than stopping at the first, and rescanning the event list for each
    # one would be needlessly quadratic.
    sent_by_uid: dict[int, list[tuple[float, int, int]]] = {}
    delivered_uids: set[int] = set()
    for e in events:
        msg = getattr(e, "message", None)
        if msg is None:
            continue
        if isinstance(e, MessageSentEvent):
            sent_by_uid.setdefault(msg.uid, []).append((e.time, e.source, e.destination))
        elif isinstance(e, MessageDeliveredEvent):
            delivered_uids.add(msg.uid)
    for hops in sent_by_uid.values():
        hops.sort()

    selected = []
    for id_range, preferred_nodes, reference_time in region_specs:
        uts_in_range = sorted(ut for ut in by_ut_uids if ut in id_range)
        if not uts_in_range:
            print(f"WARNING: no terminal with a delivered flood message found in range {id_range}.")
            continue

        fallback = None
        chosen = None
        candidates: list[tuple[float, str, int]] = []
        for ut in uts_in_range:
            for uid, label in sorted(by_ut_uids[ut]):
                if uid not in delivered_uids:
                    continue
                if fallback is None:
                    fallback = (label, uid)
                if not preferred_nodes:
                    continue
                hops = sent_by_uid.get(uid, [])
                if not hops:
                    continue
                nodes_in_path = {n for _, s, d in hops for n in (s, d)}
                if nodes_in_path & preferred_nodes:
                    candidates.append((hops[0][0], label, uid))

        if candidates:
            if reference_time is None:
                # Fallback order: first terminal in the range, first usable UID.
                chosen = (candidates[0][1], candidates[0][2])
            else:
                _, label, uid = min(candidates, key=lambda c: abs(c[0] - reference_time))
                chosen = (label, uid)

        if chosen is not None:
            selected.append(chosen)
        elif fallback is not None:
            if preferred_nodes:
                print(f"WARNING: no flood message in range {id_range} intersects the satellites of "
                      f"the legitimate traffic. Falling back to the first available terminal.")
            selected.append(fallback)
        else:
            print(f"WARNING: no delivered flood message found in range {id_range}.")

    return selected


def is_delivered_uid(events: list, uid: int) -> bool:
    return any(
        isinstance(e, MessageDeliveredEvent) and getattr(e, "message", None) is not None
        and e.message.uid == uid
        for e in events
    )


def extract_path_by_uid(events: list, uid: int) -> list[tuple[int, int]]:
    """As the shared extract_path, but filtering by message UID rather than by label.

    Flood messages need this because their textual label is not unique across terminals.
    """
    sent_events = [
        e for e in events
        if isinstance(e, MessageSentEvent) and getattr(e, "message", None) is not None
        and e.message.uid == uid
    ]
    sent_events.sort(key=lambda e: e.time)
    return [(e.source, e.destination) for e in sent_events]


class OffsetThreeColorEarthVisualizer(ThreeColorEarthVisualizer):
    """Combines the three-colour visualizer with the shared-link tube offsetting.

    Neither part is new: the visualizer already exists for this topology, and the offsetting
    technique already exists for the path renderer. Only their combination is defined here.
    """

    def __init__(self, *args, highlight_groups: list[tuple[list[tuple[int, int]], tuple[float, float, float]]] = (),
                 link_offset_scale: float = 3.0, **kwargs):
        super().__init__(*args, **kwargs)  # highlight_links stays empty by default in the base class
        self.highlight_groups = list(highlight_groups)
        self.link_offset_scale = link_offset_scale

    def build_highlight_tubes_mesh(self, satellites, groups, radius):
        flat = []
        for links, color in groups:
            for id_left, id_right in links:
                flat.append((frozenset((id_left, id_right)), id_left, id_right, color))

        link_counts: dict[frozenset, int] = {}
        for key, _, _, _ in flat:
            link_counts[key] = link_counts.get(key, 0) + 1

        seen_so_far: dict[frozenset, int] = {}
        tubes = []
        for key, id_left, id_right, color in flat:
            left = satellites.by_id(id_left)
            right = satellites.by_id(id_right)
            if left is None or right is None:
                continue
            p1 = (left.position * self.space_scale) + (left.orbital_center.position * self.interplanetary_scale)
            p2 = (right.position * self.space_scale) + (right.orbital_center.position * self.interplanetary_scale)

            n_sharing = link_counts[key]
            if n_sharing > 1:
                idx = seen_so_far.get(key, 0)
                seen_so_far[key] = idx + 1
                step = (idx - (n_sharing - 1) / 2.0) * radius * self.link_offset_scale
                offset_vec = self._perpendicular_offset_axis(p1, p2) * step
                p1 = p1 + offset_vec
                p2 = p2 + offset_vec

            transform, length = _tube_transform_for_segment(p1, p2)
            if length <= 0.0:
                continue
            cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=10)
            cylinder.apply_transform(transform)
            cylinder.visual.vertex_colors = (*color, 1.0)
            tubes.append(cylinder)

        if not tubes:
            return None
        combined = trimesh.util.concatenate(tubes)
        return pyrender.Mesh.from_trimesh(combined, smooth=False)

    @staticmethod
    def _perpendicular_offset_axis(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        d = p2 - p1
        d_norm = d / (np.linalg.norm(d) + 1e-9)
        mid = (p1 + p2) / 2.0
        radial = mid / (np.linalg.norm(mid) + 1e-9)
        axis = np.cross(d_norm, radial)
        norm = np.linalg.norm(axis)
        if norm < 1e-6:
            axis = np.cross(d_norm, np.array([0.0, 0.0, 1.0]))
            norm = np.linalg.norm(axis)
            if norm < 1e-6:
                return np.array([1.0, 0.0, 0.0])
        return axis / norm


def build_visualizer(viewport_size, time_scale, space_scale, white_bg, highlight_groups,
                      walker_scale: int = 1, ground_color=(0.0, 1.0, 0.0), ut_color=(0.0, 0.4, 1.0)):
    earth_materials = (
        f"{ASSETS_DIR}/2k_earth_daymap.jpg",
        f"{ASSETS_DIR}/2k_earth_normal_map.jpg",
        f"{ASSETS_DIR}/2k_earth_metallic_roughness_map.jpg",
    )
    bg_color = (1.0, 1.0, 1.0) if white_bg else (0.0, 0.0, 0.0)

    constellation = IridiumUserTerminalMultiConstellation(iridium_kwargs=dict(
        num_planes=6,
        sats_per_plane=11 * walker_scale,
    ))
    return OffsetThreeColorEarthVisualizer(
        constellation,
        time_scale=time_scale,
        space_scale=space_scale,
        earth_materials=earth_materials,
        viewport_size=viewport_size,
        bg_color=bg_color,
        ground_color=ground_color,
        ut_color=ut_color,
        highlight_groups=highlight_groups,
    )


def build_highlight_groups(events: list, n_legit_per_stream: int):
    """Returns the highlight groups and the render time.

    The render time is the mean of the first-hop instants of all highlighted messages, and is passed
    to update_simulation() instead of a fixed zero, because the
    satellites would otherwise be drawn where they were at time zero rather than at the moment the
    hop actually occurred."""
    legit_labels = discover_legit_labels(events, n_per_stream=n_legit_per_stream)
    if not legit_labels:
        print("WARNING: no delivered legitimate message found inside the attack window.")

    groups = []
    hop_times = []
    # Nodes touched by the legitimate path, kept per flow (0 for Fucino, 1 for Tempe), used to
    # Prefer flood messages that genuinely share a satellite with the legitimate traffic, rather
    # than a terminal selected by ID alone.
    legit_nodes_by_stream: dict[str, set[int]] = {"0": set(), "1": set()}
    # First-hop time per flow, used as the temporal reference when choosing the flood example, so
    # the highlighted paths stay clustered around a single instant that render_time can represent.
    legit_first_time_by_stream: dict[str, float] = {}

    for i, label in enumerate(legit_labels):
        color = vmp.PATH_COLOR_PALETTE[i % len(vmp.PATH_COLOR_PALETTE)]
        path = vmp.extract_path(events, label)
        vmp.print_path_summary(events, label, path, color)
        if path:
            groups.append((path, color))
            t0 = first_sent_time(events, label)
            if t0 is not None:
                hop_times.append(t0)
                m_stream = LEGIT_LABEL_RE.match(label)
                if m_stream:
                    legit_first_time_by_stream.setdefault(m_stream.group(1), t0)
            m = LEGIT_LABEL_RE.match(label)
            if m:
                # Satellites only: ground nodes, Svalbard above all, would make the intersection
                # trivially true.
                legit_nodes_by_stream[m.group(1)].update(
                    n for hop in path for n in hop if n >= FIRST_SATELLITE_ID
                )

    flood_examples = discover_flood_examples(events, region_specs=[
        (RUSSIA_EUROPE_UT_RANGE, frozenset(legit_nodes_by_stream["0"]),
         legit_first_time_by_stream.get("0")),
        (USA_UT_RANGE, frozenset(legit_nodes_by_stream["1"]),
         legit_first_time_by_stream.get("1")),
    ])
    if not flood_examples:
        print("WARNING: no delivered flood message found.")

    for label, uid in flood_examples:
        path = extract_path_by_uid(events, uid)
        print(f"{label} [uid={uid}] (RGB {FLOOD_COLOR}): {len(path)} hops, "
              f"path: {path[0][0] if path else '?'}" + "".join(f" -> {d}" for _, d in path))
        if path:
            groups.append((path, FLOOD_COLOR))
            sent_times = [
                e.time for e in events
                if isinstance(e, MessageSentEvent) and getattr(e, "message", None) is not None
                and e.message.uid == uid
            ]
            if sent_times:
                hop_times.append(min(sent_times))

    render_time = sum(hop_times) / len(hop_times) if hop_times else 0.0
    return groups, render_time


def run_interactive(viewport_size, time_scale, space_scale, white_bg, highlight_groups, render_time=0.0,
                     walker_scale=1, ground_color=(0.0, 1.0, 0.0), ut_color=(0.0, 0.4, 1.0)):
    visualizer = build_visualizer(viewport_size, time_scale, space_scale, white_bg, highlight_groups,
                                   walker_scale, ground_color, ut_color)
    visualizer.update_simulation(render_time)
    visualizer.run_simulation()
    rebuild_fn = lambda: build_visualizer(
        viewport_size, time_scale, space_scale, white_bg, highlight_groups, walker_scale, ground_color, ut_color
    )
    run_viewer_with_save_dir(visualizer, OUT_DIR, prefix="ut_flood_paths", rebuild_fn=rebuild_fn)

    while visualizer.viewer is None or visualizer.viewer.is_active:
        time.sleep(0.1)


def run_offscreen_test(viewport_size, time_scale, space_scale, white_bg, output_file, highlight_groups,
                        render_time=0.0, walker_scale=1, ground_color=(0.0, 1.0, 0.0), ut_color=(0.0, 0.4, 1.0)):
    os.makedirs(OUT_DIR, exist_ok=True)

    visualizer = build_visualizer(viewport_size, time_scale, space_scale, white_bg, highlight_groups,
                                   walker_scale, ground_color, ut_color)
    visualizer.update_simulation(render_time)

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
        description="Renders where flood messages from the compromised terminals meet the same "
                    "satellites as the legitimate Fucino and Tempe traffic into Svalbard "
                    "(attack window [990, 1080] s)."
    )
    parser.add_argument("--pickle-path", type=str, default=DEFAULT_PICKLE_PATH,
                         help="Message path pickle to render.")
    parser.add_argument("--n-legit-per-stream", type=int, default=1,
                         help="How many legitimate examples to show per flow.")
    parser.add_argument("--walker-scale", type=int, default=1)
    parser.add_argument("--time-scale", type=float, default=100.0)
    parser.add_argument("--space-scale", type=float, default=1e-6)
    parser.add_argument("--viewport-size", type=int, nargs=2, default=(800, 600))
    parser.add_argument("-w", "--white-bg", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--output-file", type=str, default="ut_flood_convergence_test.png")
    parser.add_argument("--ground-color", type=float, nargs=3, default=(0.0, 1.0, 0.0))
    parser.add_argument("--ut-color", type=float, nargs=3, default=(0.0, 0.4, 1.0))
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    viewport_size = tuple(args.viewport_size)

    events = vmp.load_events(args.pickle_path)
    highlight_groups, render_time = build_highlight_groups(events, args.n_legit_per_stream)
    print(f"Render time used to position the satellites: t={render_time:.1f}s "
          f"(mean first-hop instant of the highlighted paths, not zero)")

    ground_color = tuple(args.ground_color)
    ut_color = tuple(args.ut_color)

    if args.test:
        run_offscreen_test(viewport_size, args.time_scale, args.space_scale, args.white_bg,
                            args.output_file, highlight_groups, render_time, args.walker_scale, ground_color, ut_color)
    else:
        run_interactive(viewport_size, args.time_scale, args.space_scale, args.white_bg,
                         highlight_groups, render_time, args.walker_scale, ground_color, ut_color)
