"""Graph analysis identifying the minimum cut between two gateways.

A purely topological analysis, with no simulation involved, used to derive the parameters the
attack scripts need: which satellites to target, and in which time windows to expect which effect.

The relation under study runs from Fucino, ground station 5 in Italy, to Tempe, ground station 0 in
Arizona, on the reconstructed Iridium NEXT model with the same ISL-only constraint applied to
intermediate hops, expressed here in networkx rather than as a routing override.

It computes:

  1. The shortest path in hops from Fucino to Tempe at the start of the run, respecting the
     ISL-only constraint.
  2. The minimum cut between the two, in edges and in nodes: the smallest set whose removal
     disconnects them at that instant.
  3. A variant in which the direct uplinks of both stations are excluded from the cut, to measure
     what an attack further from the target would cost.
  4. The visibility schedule of the two attacked satellites across the run, partitioned into the
     three categories the results are later reported by: intervals in which every satellite Fucino
     can reach is attacked, intervals in which an alternative exists, and intervals in which
     neither attacked satellite is visible.
"""

import json
import os
import sys

import networkx as nx

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "model_and_baseline"),
    os.path.dirname(os.path.abspath(__file__)),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from presets_reconstructed import IridiumReconstructedMultiConstellation  # noqa: E402

FUCINO = 5     # Fucino, Italy: the European gateway
TEMPE = 0      # Tempe, Arizona (USA)
ATTACKED_NODES = {9, 26}   # the only two satellites Fucino can reach at the start of the run

SIM_MAX_TIME = 6000.0
TIMESTEP = 15.0   # the same timestep used by the simulations


def build_static_graph(constellation, t: float = 0.0, ground_nodes=(FUCINO, TEMPE)):
    """ISL-only graph plus the uplinks of the given ground stations alone.

    Reproduces exactly the constraint applied at runtime by build_ground_to_ground_override():
    intermediate hops are inter-satellite links, and no ground station can act as a relay.
    """
    constellation.update(t)
    G = nx.Graph()
    for s1, s2 in constellation.isls:
        G.add_edge(s1, s2, kind="ISL")
    for a, b in constellation.ills:
        if a in ground_nodes or b in ground_nodes:
            G.add_edge(a, b, kind="ILL")
    return G


def min_cuts_at_t0(constellation):
    G = build_static_graph(constellation, t=0.0)

    sp = nx.shortest_path(G, FUCINO, TEMPE)
    edge_cut = nx.minimum_edge_cut(G, FUCINO, TEMPE)
    node_cut = nx.minimum_node_cut(G, FUCINO, TEMPE)

    print(f"Fucino({FUCINO}) uplinks to satellites: {sorted(G.neighbors(FUCINO))}")
    print(f"Tempe({TEMPE})  uplinks to satellites: {sorted(G.neighbors(TEMPE))}")
    print(f"\nShortest path (ISL-only) Fucino->Tempe a t=0: {sp}  ({len(sp) - 1} hop)")
    print(f"MIN EDGE CUT: {len(edge_cut)} archi -> {sorted(edge_cut)}")
    print(f"MIN NODE CUT: {len(node_cut)} nodi -> {sorted(node_cut)}")

    assert node_cut == ATTACKED_NODES, f"sanity check fallito: min node cut = {node_cut}, atteso {ATTACKED_NODES}"

    return sp, edge_cut, node_cut


def dorsale_protetta_cut(constellation):
    """Scenario B (backlog): min-cut vietando di toccare gli uplink diretti di Fucino/Tempe -
    what an attack further out, on the ISL backbone rather than on the local uplink, would cost."""
    G = build_static_graph(constellation, t=0.0)
    Gc = nx.Graph()
    Gc.add_nodes_from(G.nodes)
    for a, b in G.edges():
        capacity = 1000 if (FUCINO in (a, b) or TEMPE in (a, b)) else 1
        Gc.add_edge(a, b, capacity=capacity)
    cut_val, (S, T) = nx.minimum_cut(Gc, FUCINO, TEMPE)
    cut_edges = sorted((u, v) for u in S for v in G.neighbors(u) if v in T)
    print(f"\n[Scenario B, backlog] min-cut sulla dorsale (uplink protetti): {cut_val} archi -> {cut_edges}")
    return cut_edges


def fucino_ill_schedule(constellation, dt: float = TIMESTEP, t_max: float = SIM_MAX_TIME):
    """Steps the constellation forward and records which satellites Fucino can reach.

    The per-instant record is then compressed into intervals of (start, end, satellites).
    """
    schedule = []
    t = 0.0
    while t <= t_max:
        constellation.update(t)
        neighbors = frozenset(
            b if a == FUCINO else a for a, b in constellation.ills if FUCINO in (a, b)
        )
        schedule.append((t, neighbors))
        t += dt

    intervals = []
    cur_sats, cur_start = schedule[0][1], schedule[0][0]
    for t, sats in schedule[1:]:
        if sats != cur_sats:
            intervals.append((cur_start, t, cur_sats))
            cur_start, cur_sats = t, sats
    intervals.append((cur_start, schedule[-1][0] + dt, cur_sats))
    return intervals


def classify_windows(intervals, attacked=ATTACKED_NODES):
    """Classifies each of Fucino's visibility intervals into one of three categories.

    - blackout: every visible satellite belongs to the attacked set, so the graph itself guarantees
      that no path exists during that window.
    - contested: the attacked satellites are visible alongside at least one that is not, so routing
      may find an alternative. Whether it does cannot be deduced here and needs a simulation.
    - unaffected: neither attacked satellite is visible at that instant.
    """
    blackout, conteso, non_toccato = [], [], []
    for start, end, sats in intervals:
        if sats and sats <= attacked:
            blackout.append((start, end))
        elif sats & attacked:
            conteso.append((start, end))
        else:
            non_toccato.append((start, end))
    return {"blackout": blackout, "conteso": conteso, "non_toccato": non_toccato}


def main():
    constellation = IridiumReconstructedMultiConstellation(iridium_kwargs=dict(num_planes=6, sats_per_plane=11))

    sp, edge_cut, node_cut = min_cuts_at_t0(constellation)
    dorsale_protetta_cut(constellation)

    intervals = fucino_ill_schedule(constellation)
    windows = classify_windows(intervals)

    def total(ws):
        return sum(e - s for s, e in ws)

    print(f"\n{len(intervals)} distinct visibility intervals for Fucino over {SIM_MAX_TIME:.0f}s\n")
    for label, ws in windows.items():
        print(f"  {label:12s}: {total(ws):6.0f}s ({100 * total(ws) / SIM_MAX_TIME:5.1f}%)  -> {ws}")

    output = {
        "fucino": FUCINO,
        "tempe": TEMPE,
        "attacked_nodes": sorted(ATTACKED_NODES),
        "shortest_path_t0": sp,
        "min_edge_cut_t0": sorted(list(edge_cut)),
        "min_node_cut_t0": sorted(list(node_cut)),
        "windows": {k: v for k, v in windows.items()},
    }
    out_path = os.path.join(os.path.dirname(__file__), "min_cut_analysis_result.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OK] Results saved to {out_path}")


if __name__ == "__main__":
    main()
