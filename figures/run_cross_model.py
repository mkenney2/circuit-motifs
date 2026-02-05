"""Run motif analysis on cross-model Dallas graphs and compare with original."""

import sys
import json
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_loader import load_attribution_graph
from src.motif_census import (
    compute_motif_census, find_motif_instances,
    MOTIF_FFL, TRIAD_LABELS,
)
from src.null_model import generate_configuration_null

RESULTS_PATH = PROJECT_ROOT / "data" / "results"

# Configs: (label, path, threshold)
CONFIGS = [
    ("Haiku-CLT (original)", "data/raw/multihop/capital-state-dallas.json", 1.0),
    ("Gemma-2-2B (t=15)", "data/raw/cross_model/dallas-capital-gemma2-2b.json", 15.0),
    ("Gemma-2-2B (t=10)", "data/raw/cross_model/dallas-capital-gemma2-2b.json", 10.0),
    ("Qwen3-4B (t=5)", "data/raw/cross_model/dallas-capital-qwen3-4b.json", 5.0),
    ("Qwen3-4B (t=3)", "data/raw/cross_model/dallas-capital-qwen3-4b.json", 3.0),
]

N_RANDOM = 100
KEY_MOTIFS = ["030T", "021C", "111U", "021D", "021U", "030C"]


def run_analysis():
    results = {}

    for label, path, threshold in CONFIGS:
        full_path = PROJECT_ROOT / path
        print(f"\n{'='*60}")
        print(f"{label} (threshold={threshold})")
        print(f"{'='*60}")

        g = load_attribution_graph(str(full_path), weight_threshold=threshold)
        print(f"  Graph: {g.vcount()} nodes, {g.ecount()} edges")

        if g.vcount() < 10 or g.ecount() < 10:
            print("  SKIP: too small")
            continue

        # Motif census
        census = compute_motif_census(g, size=3)
        connected = census.connected_counts()
        print(f"  Raw counts: FFL={connected.get('030T',0)}, "
              f"Chain={connected.get('021C',0)}, "
              f"Fan-in={connected.get('021U',0)}, "
              f"Fan-out={connected.get('021D',0)}")

        # Top FFL instances
        ffl_instances = find_motif_instances(g, MOTIF_FFL, size=3, max_instances=10)
        print(f"  FFL instances found: {len(ffl_instances)}")
        for i, inst in enumerate(ffl_instances[:3]):
            roles = []
            for nid, role in inst.node_roles.items():
                clerp = g.vs[nid]["clerp"] if "clerp" in g.vs.attributes() else ""
                layer = g.vs[nid]["layer"]
                roles.append(f"{role}=L{layer}'{clerp[:25]}'")
            print(f"    #{i}: w={inst.total_weight:.1f} | {' | '.join(roles)}")

        # Null model
        print(f"  Running null model ({N_RANDOM} iterations)...")
        t0 = time.time()
        null_result = generate_configuration_null(
            g, n_random=N_RANDOM, motif_size=3, show_progress=True
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        # Results
        print(f"  Z-scores and SP values:")
        for motif in KEY_MOTIFS:
            idx = TRIAD_LABELS.index(motif)
            z = null_result.z_scores[idx]
            sp = null_result.significance_profile[idx]
            print(f"    {motif:6s}: Z={z:+8.1f}  SP={sp:+.4f}")

        results[label] = {
            "z_scores": {
                TRIAD_LABELS[i]: float(null_result.z_scores[i])
                for i in range(16)
            },
            "significance_profile": {
                TRIAD_LABELS[i]: float(null_result.significance_profile[i])
                for i in range(16)
            },
            "n_nodes": g.vcount(),
            "n_edges": g.ecount(),
            "threshold": threshold,
            "path": path,
        }

    # Save
    out_path = RESULTS_PATH / "cross_model_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY: SP values across models")
    print(f"{'='*60}")
    header = f"{'Model':<25s}" + "".join(f"{m:>10s}" for m in KEY_MOTIFS)
    print(header)
    print("-" * len(header))
    for label in results:
        sp = results[label]["significance_profile"]
        row = f"{label:<25s}"
        for m in KEY_MOTIFS:
            row += f"{sp.get(m, 0):>+10.3f}"
        n = results[label]["n_nodes"]
        e = results[label]["n_edges"]
        row += f"  ({n}n/{e}e)"
        print(row)

    # Cosine similarities
    labels = list(results.keys())
    print(f"\nCosine similarity matrix:")
    motifs_all = [m for m in TRIAD_LABELS if m != "003"]
    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            if j <= i:
                continue
            v1 = np.array([results[l1]["significance_profile"].get(m, 0) for m in motifs_all])
            v2 = np.array([results[l2]["significance_profile"].get(m, 0) for m in motifs_all])
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            print(f"  {l1} vs {l2}: {cos:.4f}")


if __name__ == "__main__":
    run_analysis()
