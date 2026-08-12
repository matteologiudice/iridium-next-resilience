"""Renderer variant that separates 3D tubes drawn along a shared link.

When two or more highlighted paths traverse the same physical hop, for instance when two flows
share their final hop into the same ground station, their tubes are drawn at exactly the same
position in space and one colour hides the other completely. The result looks as though a path
stops at a satellite, when it is only concealed beneath the tube of another path crossing the same
link. The data are unaffected: every path reaches its destination with the expected receive and
deliver events, and this is purely a rendering artefact.

Each tube on a shared link is therefore offset laterally from the centre line, perpendicular to the
Earth's surface at that point, using the radial direction from the Earth's centre as the reference
so the offset runs along the surface rather than sinking into or lifting off the globe. N paths
sharing a link are drawn as N parallel tubes. Links that are not shared are drawn exactly as
before, with no offset applied.

All of the surrounding logic (loading the pickle, extracting paths, the command line) is imported
from visualize_message_path rather than duplicated. The only difference is the visualizer class,
replaced by a subclass of TwoColorEarthVisualizer that overrides build_highlight_tubes_mesh alone.
"""

import os
import sys

import numpy as np
import trimesh
import trimesh.creation
import pyrender

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "model_and_baseline"),
    os.path.dirname(os.path.abspath(__file__)),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import visualize_message_path as vmp  # noqa: E402  (riuso completo: load_events, extract_path, CLI, ecc.)
from visualize_iridium_earth import (  # noqa: E402
    TwoColorEarthVisualizer,
    _tube_transform_for_segment,
    run_viewer_with_save_dir,
    ASSETS_DIR,
    SCREENSHOT_SUPERSAMPLE,
)


class OffsetTwoColorEarthVisualizer(TwoColorEarthVisualizer):
    """As TwoColorEarthVisualizer, except that build_highlight_tubes_mesh offsets shared links.

    A link highlighted by more than one path is drawn as several laterally offset tubes rather than
    as tubes stacked exactly on top of one another.
    """

    def __init__(self, *args, link_offset_scale: float = 3.0, **kwargs):
        """link_offset_scale: distanza tra tubi sovrapposti, in multipli di highlight_radius -
        A value of 3.0 separates them clearly while keeping them close and parallel."""
        super().__init__(*args, **kwargs)
        self.link_offset_scale = link_offset_scale

    def build_highlight_tubes_mesh(self, satellites, groups, radius):
        # Step 1: flatten (link, colour) for every group and count how many groups use each
        # physical link. The key is unordered, so (6,9) and (9,6) are the same link.
        flat = []  # list of (frozenset({a, b}), id_left, id_right, colour)
        for links, color in groups:
            for id_left, id_right in links:
                flat.append((frozenset((id_left, id_right)), id_left, id_right, color))

        link_counts = {}
        for key, _, _, _ in flat:
            link_counts[key] = link_counts.get(key, 0) + 1

        # Running index for each occurrence of a shared link, used to compute a symmetric offset:
        # two occurrences give [-0.5, +0.5] steps, three give [-1, 0, +1], and so on.
        seen_so_far = {}

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
                # Offset simmetrico attorno alla linea centrale: per n_sharing occorrenze,
                # step in {-(n-1)/2, ..., +(n-1)/2} * (radius * link_offset_scale)
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
        """Axis perpendicular to the segment p1 to p2, running along the Earth's surface.

        The radial direction from the Earth's centre to the segment's midpoint is used as the
        reference for the cross product, so the resulting offset is tangent to the sphere at that
        point rather than sinking into or lifting off the globe.
        """
        d = p2 - p1
        d_norm = d / (np.linalg.norm(d) + 1e-9)
        mid = (p1 + p2) / 2.0
        radial = mid / (np.linalg.norm(mid) + 1e-9)
        axis = np.cross(d_norm, radial)
        norm = np.linalg.norm(axis)
        if norm < 1e-6:
            # d quasi parallelo alla radiale (raro, es. link quasi-verticale rispetto al globo) -
            # fallback a una perpendicolare arbitraria.
            axis = np.cross(d_norm, np.array([0.0, 0.0, 1.0]))
            norm = np.linalg.norm(axis)
            if norm < 1e-6:
                return np.array([1.0, 0.0, 0.0])
        return axis / norm


def build_message_path_visualizer(viewport_size, time_scale, space_scale, white_bg, highlight_groups,
                                   walker_scale: int = 1, ground_color=(0.0, 1.0, 0.0)):
    """As visualize_message_path.build_message_path_visualizer, with the offsetting subclass."""
    earth_materials = (
        f"{ASSETS_DIR}/2k_earth_daymap.jpg",
        f"{ASSETS_DIR}/2k_earth_normal_map.jpg",
        f"{ASSETS_DIR}/2k_earth_metallic_roughness_map.jpg",
    )
    bg_color = (1.0, 1.0, 1.0) if white_bg else (0.0, 0.0, 0.0)

    constellation = vmp.IridiumReconstructedMultiConstellation(iridium_kwargs=dict(
        num_planes=6,
        sats_per_plane=11 * walker_scale,
    ))
    return OffsetTwoColorEarthVisualizer(
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
    run_viewer_with_save_dir(visualizer, vmp.OUT_DIR, prefix="message_path_iridium_g2g_offset", rebuild_fn=rebuild_fn)

    import time
    while visualizer.viewer is None or visualizer.viewer.is_active:
        time.sleep(0.1)


def run_offscreen_test(viewport_size, time_scale, space_scale, white_bg, output_file, highlight_groups, walker_scale=1,
                        ground_color=(0.0, 1.0, 0.0)):
    os.makedirs(vmp.OUT_DIR, exist_ok=True)

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

    out_path = os.path.join(vmp.OUT_DIR, output_file)
    import PIL.Image
    PIL.Image.fromarray(color).save(out_path)
    print(f"[OK] Offscreen screenshot saved to {out_path} ({color.shape[1]}x{color.shape[0]})")


if __name__ == "__main__":
    args = vmp.get_parser().parse_args()
    viewport_size = tuple(args.viewport_size)
    pickle_path = args.pickle_path or vmp.DEFAULT_PICKLE_PATH

    events = vmp.load_events(pickle_path)
    message_labels = args.message_labels or vmp.discover_default_message_labels(events)
    if not message_labels:
        raise ValueError(f"No message found in pickle {pickle_path!r}")

    highlight_groups = []
    for i, label in enumerate(message_labels):
        color = vmp.PATH_COLOR_PALETTE[i % len(vmp.PATH_COLOR_PALETTE)]
        path = vmp.extract_path(events, label)
        vmp.print_path_summary(events, label, path, color)
        if path:
            highlight_groups.append((path, color))

    ground_color = tuple(args.ground_color)

    if args.test:
        run_offscreen_test(viewport_size, args.time_scale, args.space_scale, args.white_bg,
                            args.output_file, highlight_groups, args.walker_scale, ground_color)
    else:
        run_interactive(viewport_size, args.time_scale, args.space_scale, args.white_bg,
                         highlight_groups, args.walker_scale, ground_color)
