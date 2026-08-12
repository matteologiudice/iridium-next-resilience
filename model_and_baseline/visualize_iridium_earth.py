"""Three-dimensional rendering of the reconstructed Iridium NEXT constellation.

Renders the 66 satellites together with the eight reconstructed ground stations, over an Earth
texture, either in an interactive viewer or as an offscreen screenshot.

Two node colours
----------------
The simulator's own EarthVisualizer draws every node, satellites and ground stations alike, as
identical red spheres. With only eight ground stations rather than the 256 synthetic defaults,
telling the two apart visually is far more useful, particularly for checking the ground-to-ground,
ISL-only scenarios by eye.

TwoColorEarthVisualizer below is a subclass of dsns.visualizer.MultiConstellationVisualizer that
builds two separate meshes instead of one, calling build_nodes_mesh_from_sats() once on
multi_constellation.ground_constellation.satellites and once on
multi_constellation.iridium_constellation.satellites. Those two attributes already exist on
IridiumReconstructedMultiConstellation, so no manual classification by node ID is needed. No DSNS
method is modified; the library class is only subclassed.

Ground stations are drawn in green by default. Green does carry some risk of blending with the
land masses of the Earth daymap texture, which are themselves green and brown, so the colour is
configurable through --ground-color if a stronger contrast is wanted.

Modes
-----
  default   Interactive pyrender viewer. The mouse orbits and zooms, S takes a screenshot without
            opening a dialog, and R starts and stops GIF recording, also without a dialog.
  --test    Offscreen render, saving a PNG under results/renders without opening a window.

--highlight-link "a,b" may be repeated to draw specific links as coloured 3D tubes, yellow by
default and configurable through --highlight-color. Highlighting and the two-colour node meshes are
independent responsibilities and coexist in the same class without conflict.
"""

import argparse
import datetime
import os
import sys
import time

import imageio
import numpy as np
import pyrender
import trimesh
import trimesh.creation
import trimesh.transformations

import dsns
from dsns.visualizer import MultiConstellationVisualizer
from dsns.helpers import EARTH_RADIUS, EARTH_ROTATION_PERIOD

# The repository is a set of flat scripts rather than a package, so the repository root is put on
# the path explicitly to import presets_reconstructed, which lives one level above this file.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "model_and_baseline"),
    os.path.dirname(os.path.abspath(__file__)),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from presets_reconstructed import IridiumReconstructedMultiConstellation  # noqa: E402

ASSETS_DIR = os.environ.get("DSNS_ASSETS_DIR", os.path.join(os.path.dirname(dsns.__file__), "examples", "assets"))
OUT_DIR = os.path.join(_REPO_ROOT, "results", "renders")

# Screenshots, taken with S or with --test, are rendered offscreen at SCREENSHOT_SUPERSAMPLE times
# the resolution of the interactive window, to avoid the coarse look of a small PNG.
SCREENSHOT_SUPERSAMPLE = 3


def _tube_transform_for_segment(p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, float]:
    """Returns the 4x4 transform placing a unit cylinder along the segment from p1 to p2.

    The cylinder is assumed to lie on the Z axis, centred on the origin, with height 1. Returns the
    transform together with the segment length.
    """
    direction = p2 - p1
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        return np.eye(4), 0.0
    direction_norm = direction / length
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, direction_norm)
    axis_norm = np.linalg.norm(axis)
    dot = float(np.clip(np.dot(z_axis, direction_norm), -1.0, 1.0))
    if axis_norm < 1e-9:
        rotation = np.eye(4) if dot > 0 else trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    else:
        angle = np.arccos(dot)
        rotation = trimesh.transformations.rotation_matrix(angle, axis)
    midpoint = (p1 + p2) / 2.0
    translation = trimesh.transformations.translation_matrix(midpoint)
    return translation @ rotation, length


class TwoColorEarthVisualizer(MultiConstellationVisualizer):
    """Renders satellites and ground stations as separate meshes with distinct colours.

    Equivalent to dsns.presets.EarthVisualizer except that nodes are not drawn as one uniform mesh.
    Also supports highlighting specific links as coloured 3D tubes.
    """

    def __init__(
        self,
        multi_constellation: IridiumReconstructedMultiConstellation,
        time_scale: float = 100.0,
        space_scale: float = 1e-6,
        earth_materials: tuple[str, str, str] | None = None,
        viewport_size: tuple[int, int] = (800, 600),
        bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        sat_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
        ground_color: tuple[float, float, float] = (0.0, 1.0, 0.0),  # green, configurable
        sat_radius: float = 0.08,
        ground_radius: float = 0.16,  # larger than the satellite spheres: there are only eight of
                                      # them, and they should stand out despite being stationary
        isl_color: tuple[float, float, float] = (0.0, 1.0, 0.0),
        ill_color: tuple[float, float, float] = (1.0, 0.0, 1.0),
        highlight_links: list[tuple[int, int]] = (),
        highlight_color: tuple[float, float, float] = (1.0, 1.0, 0.0),
        highlight_groups: list[tuple[list[tuple[int, int]], tuple[float, float, float]]] = None,
        highlight_radius: float = 0.045,
    ):
        super().__init__(
            multi_constellation,
            time_scale=time_scale,
            space_scale=space_scale,
            viewport_size=viewport_size,
            sat_color=sat_color,
            isl_color=isl_color,
            ill_color=ill_color,
            bg_color=bg_color,
        )

        earth_color = (107 / 255, 147 / 255, 214 / 255)
        self.add_planet(
            radius=EARTH_RADIUS,
            rotation_period=EARTH_ROTATION_PERIOD,
            materials=earth_materials,
            color=earth_color,
        )

        # Overrides self.sat_mesh, built by the parent with the default radius and colour, and adds
        # a second mesh dedicated to the ground stations.
        self.sat_mesh = trimesh.creation.icosphere(radius=sat_radius, subdivisions=1)
        self.sat_mesh.visual.vertex_colors = sat_color  # type: ignore

        self.ground_mesh = trimesh.creation.icosphere(radius=ground_radius, subdivisions=1)
        self.ground_mesh.visual.vertex_colors = ground_color  # type: ignore

        self.ground_node = None

        if highlight_groups is not None:
            self.highlight_groups = [(list(links), color) for links, color in highlight_groups]
        else:
            self.highlight_groups = [(list(highlight_links), highlight_color)] if highlight_links else []
        self.highlight_radius = highlight_radius
        self.highlight_node = None

    def build_highlight_tubes_mesh(self, satellites, groups, radius):
        tubes = []
        for links, color in groups:
            for id_left, id_right in links:
                left = satellites.by_id(id_left)
                right = satellites.by_id(id_right)
                if left is None or right is None:
                    continue
                p1 = (left.position * self.space_scale) + (left.orbital_center.position * self.interplanetary_scale)
                p2 = (right.position * self.space_scale) + (right.orbital_center.position * self.interplanetary_scale)
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

    def update_simulation(self, t: float):
        self.multiconstellation.update(t)

        for (planet_node, planet_center, planet_radius, planet_rotation_period) in self.planets:
            planet_node.translation = planet_center.position * self.interplanetary_scale
            angle = (2 * np.pi * t / planet_rotation_period) + np.pi
            base_rotation = trimesh.transformations.quaternion_about_axis(np.pi, [0, 1, 0])
            angle_rotation = trimesh.transformations.quaternion_about_axis(angle, [1, 0, 0])
            planet_node.rotation = trimesh.transformations.quaternion_multiply(base_rotation, angle_rotation)

        # Ground stations: the dedicated attribute alone, not multiconstellation.satellites, which
        # would mix them back in with the satellites.
        ground_mesh_built = self.build_nodes_mesh_from_sats(
            self.ground_mesh, self.multiconstellation.ground_constellation.satellites
        )
        if self.ground_node is not None:
            self.scene.remove_node(self.ground_node)
        self.ground_node = self.scene.add(ground_mesh_built)

        # Satellites: the constellation's own attribute, kept separate from the ground stations.
        sats_mesh_built = self.build_nodes_mesh_from_sats(
            self.sat_mesh, self.multiconstellation.iridium_constellation.satellites
        )
        if self.sats_node is not None:
            self.scene.remove_node(self.sats_node)
        self.sats_node = self.scene.add(sats_mesh_built)

        # ISLs and ILLs need the full collection instead, because links reference global IDs that
        # may belong to either sub-constellation.
        isl_mesh = self.build_links_mesh(self.multiconstellation.satellites, self.multiconstellation.isls, self.isl_color)
        if self.isls_node is not None:
            self.scene.remove_node(self.isls_node)
            self.isls_node = None
        if isl_mesh is not None:
            self.isls_node = self.scene.add(isl_mesh)

        ill_mesh = self.build_links_mesh(self.multiconstellation.satellites, self.multiconstellation.ills, self.ill_color)
        if self.ills_node is not None:
            self.scene.remove_node(self.ills_node)
            self.ills_node = None
        if ill_mesh is not None:
            self.ills_node = self.scene.add(ill_mesh)

        if self.highlight_node is not None:
            self.scene.remove_node(self.highlight_node)
            self.highlight_node = None
        if self.highlight_groups:
            highlight_mesh = self.build_highlight_tubes_mesh(
                self.multiconstellation.satellites, self.highlight_groups, self.highlight_radius
            )
            if highlight_mesh is not None:
                self.highlight_node = self.scene.add(highlight_mesh)


def build_visualizer(viewport_size: tuple[int, int], time_scale: float, space_scale: float,
                      white_bg: bool, highlight_links: list[tuple[int, int]] = (),
                      highlight_color: tuple[float, float, float] = (1.0, 1.0, 0.0),
                      ground_color: tuple[float, float, float] = (0.0, 1.0, 0.0),
                      walker_scale: int = 1) -> TwoColorEarthVisualizer:
    constellation = IridiumReconstructedMultiConstellation(iridium_kwargs=dict(
        num_planes=6,
        sats_per_plane=11 * walker_scale,
    ))

    visualizer = TwoColorEarthVisualizer(
        constellation,
        time_scale=time_scale,
        space_scale=space_scale,
        earth_materials=(
            f"{ASSETS_DIR}/2k_earth_daymap.jpg",
            f"{ASSETS_DIR}/2k_earth_normal_map.jpg",
            f"{ASSETS_DIR}/2k_earth_metallic_roughness_map.jpg",
        ),
        viewport_size=viewport_size,
        bg_color=(1.0, 1.0, 1.0) if white_bg else (0.0, 0.0, 0.0),
        ground_color=ground_color,
        highlight_links=highlight_links,
        highlight_color=highlight_color,
    )
    return visualizer


def print_link_counts(visualizer: TwoColorEarthVisualizer) -> None:
    # isls and ills are only populated once update() has run, not at instantiation.
    constellation = visualizer.multiconstellation
    print(f"Reconstructed ground stations (green): {len(constellation.ground_constellation.satellites.ids)}")
    print(f"Iridium satellites (red): {len(constellation.iridium_constellation.satellites.ids)}")
    print(f"ISLs (satellite to satellite): {len(constellation.isls)}")
    print(f"ILL (ground-satellite): {len(constellation.ills)}")


def save_screenshot_no_dialog(viewer: pyrender.Viewer, save_directory: str, prefix: str = "iridium_g2g_earth",
                               rebuild_fn=None) -> None:
    """Saves a screenshot without opening a file dialog.

    The default S key opens a Tkinter dialog, which crashes the process on macOS when a pyglet
    window is already open. This bypasses it, saving directly under a timestamped name. When
    rebuild_fn is supplied, an independent twin visualizer is used to render at higher resolution.
    """
    os.makedirs(save_directory, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_directory, f"{prefix}_{timestamp}.png")

    if rebuild_fn is None:
        color = viewer._renderer.read_color_buf()
    else:
        fresh_visualizer = rebuild_fn()
        fresh_visualizer.update_simulation(0.0)

        width, height = viewer.viewport_size
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=width / height)
        camera_pose = viewer._camera_node.matrix.copy()
        fresh_visualizer.scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        fresh_visualizer.scene.add(light, pose=camera_pose)

        renderer = pyrender.OffscreenRenderer(
            viewport_width=width * SCREENSHOT_SUPERSAMPLE,
            viewport_height=height * SCREENSHOT_SUPERSAMPLE,
        )
        color, _depth = renderer.render(fresh_visualizer.scene)
        renderer.delete()

    imageio.imwrite(filename, color)
    print(f"[OK] Screenshot saved to {filename} ({color.shape[1]}x{color.shape[0]})")


def toggle_gif_recording(viewer: pyrender.Viewer, save_directory: str, prefix: str = "iridium_g2g_earth") -> None:
    """Starts and stops GIF recording without opening a file dialog, as above for screenshots."""
    os.makedirs(save_directory, exist_ok=True)
    if viewer.viewer_flags["record"]:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_directory, f"{prefix}_{timestamp}.gif")
        viewer.save_gif(filename=filename)
        viewer.set_caption(viewer.viewer_flags["window_title"])
        print(f"[OK] GIF saved to {filename}")
    else:
        viewer.set_caption(f"{viewer.viewer_flags['window_title']} (RECORDING)")
        print("Registrazione GIF avviata — premi di nuovo R per fermarla e salvarla.")
    viewer.viewer_flags["record"] = not viewer.viewer_flags["record"]


def run_viewer_with_save_dir(visualizer: TwoColorEarthVisualizer, save_directory: str, prefix: str = "iridium_g2g_earth",
                              rebuild_fn=None) -> None:
    camera = pyrender.PerspectiveCamera(
        yfov=np.pi / 3.0, aspectRatio=visualizer.viewport_size[0] / visualizer.viewport_size[1]
    )
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 20.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    visualizer.scene.add(camera, pose=camera_pose)

    os.makedirs(save_directory, exist_ok=True)
    visualizer.viewer = pyrender.Viewer(
        visualizer.scene,
        viewport_size=visualizer.viewport_size,
        use_raymond_lighting=True,
        registered_keys={
            "s": (save_screenshot_no_dialog, [save_directory, prefix, rebuild_fn]),
            "r": (toggle_gif_recording, [save_directory, prefix]),
        },
    )


def run_interactive(viewport_size, time_scale, space_scale, white_bg, highlight_links=(),
                     highlight_color=(1.0, 1.0, 0.0), ground_color=(0.0, 1.0, 0.0), walker_scale=1):
    visualizer = build_visualizer(viewport_size, time_scale, space_scale, white_bg, highlight_links,
                                   highlight_color, ground_color, walker_scale)
    visualizer.update_simulation(0.0)
    print_link_counts(visualizer)
    if highlight_links:
        print(f"Link evidenziati: {highlight_links}")
    visualizer.run_simulation()
    rebuild_fn = lambda: build_visualizer(viewport_size, time_scale, space_scale, white_bg, highlight_links,
                                           highlight_color, ground_color, walker_scale)
    run_viewer_with_save_dir(visualizer, OUT_DIR, rebuild_fn=rebuild_fn)

    while visualizer.viewer is None or visualizer.viewer.is_active:
        time.sleep(0.1)


def run_offscreen_test(viewport_size, time_scale, space_scale, white_bg, output_file: str,
                        highlight_links=(), highlight_color=(1.0, 1.0, 0.0),
                        ground_color=(0.0, 1.0, 0.0), walker_scale=1):
    """Builds the same scene as the interactive viewer and renders it offscreen.

    Used to check automatically that the Earth, satellites, ground stations, ISLs and ILLs are all
    generated correctly, without having to open and drive a window by hand.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    visualizer = build_visualizer(viewport_size, time_scale, space_scale, white_bg, highlight_links,
                                   highlight_color, ground_color, walker_scale)
    visualizer.update_simulation(0.0)
    print_link_counts(visualizer)
    if highlight_links:
        print(f"Link evidenziati: {highlight_links}")

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
        description="3D rendering of the reconstructed Iridium NEXT constellation: 66 satellites "
                    "and 8 ground stations, in distinct colours, over the Earth."
    )
    parser.add_argument("--walker-scale", type=int, default=1,
                         help="Must match the --walker-scale used by the simulation scripts, "
                              "otherwise the number of satellites shown will not match the one "
                              "that was simulated.")
    parser.add_argument("--time-scale", type=float, default=100.0, help="Speed factor (simulated seconds per real second)")
    parser.add_argument("--space-scale", type=float, default=1e-6, help="Spatial scale factor")
    parser.add_argument("--viewport-size", type=int, nargs=2, default=(800, 600))
    parser.add_argument("-w", "--white-bg", action="store_true", help="White background instead of black")
    parser.add_argument("--test", action="store_true",
                         help="Do not open the interactive window: render offscreen and save a PNG screenshot")
    parser.add_argument("--output-file", type=str, default="iridium_g2g_earth_test.png",
                         help="Name of the PNG written under results/renders (with --test only)")
    parser.add_argument("--highlight-link", type=str, action="append", default=[], dest="highlight_links",
                         help="Satellite pair 'a,b' to highlight in a colour distinct from the "
                              "ordinary green ISLs. May be repeated.")
    parser.add_argument("--highlight-color", type=float, nargs=3, default=(1.0, 1.0, 0.0),
                         metavar=("R", "G", "B"), help="RGB colour (0-1) for the highlighted links, yellow by default.")
    parser.add_argument("--ground-color", type=float, nargs=3, default=(0.0, 1.0, 0.0),
                         metavar=("R", "G", "B"), help="RGB colour (0-1) for the ground stations, green by default.")
    return parser


def parse_highlight_links(raw: list[str]) -> list[tuple[int, int]]:
    links = []
    for item in raw:
        try:
            a_str, b_str = item.split(",")
            links.append((int(a_str.strip()), int(b_str.strip())))
        except ValueError:
            raise ValueError(f"--highlight-link must have the form 'a,b' (for example '9,3'), got: {item!r}")
    return links


if __name__ == "__main__":
    args = get_parser().parse_args()
    viewport_size = tuple(args.viewport_size)
    highlight_links = parse_highlight_links(args.highlight_links)
    highlight_color = tuple(args.highlight_color)
    ground_color = tuple(args.ground_color)

    if args.test:
        run_offscreen_test(viewport_size, args.time_scale, args.space_scale, args.white_bg, args.output_file,
                            highlight_links, highlight_color, ground_color, args.walker_scale)
    else:
        run_interactive(viewport_size, args.time_scale, args.space_scale, args.white_bg,
                         highlight_links, highlight_color, ground_color, args.walker_scale)
