"""circuit-motifs: Network motif analysis of LLM attribution graphs."""

__version__ = "0.1.0"

from src.graph_loader import load_attribution_graph, graph_summary, load_graphs_from_directory
from src.motif_census import (
    compute_motif_census,
    find_motif_instances,
    MOTIF_FFL,
    MOTIF_CHAIN,
    MOTIF_FAN_IN,
    MOTIF_FAN_OUT,
    MOTIF_CYCLE,
    MOTIF_COMPLETE,
    TRIAD_LABELS,
)
from src.null_model import generate_configuration_null, NullModelResult
from src.comparison import build_task_profile, TaskProfile
from src.visualization import (
    plot_zscore_bar,
    plot_zscore_heatmap,
    plot_graph_with_motif,
    plot_top_motif,
)
from src.pipeline import run_pipeline

__all__ = [
    "__version__",
    "load_attribution_graph",
    "graph_summary",
    "load_graphs_from_directory",
    "compute_motif_census",
    "find_motif_instances",
    "MOTIF_FFL",
    "MOTIF_CHAIN",
    "MOTIF_FAN_IN",
    "MOTIF_FAN_OUT",
    "MOTIF_CYCLE",
    "MOTIF_COMPLETE",
    "TRIAD_LABELS",
    "generate_configuration_null",
    "NullModelResult",
    "build_task_profile",
    "TaskProfile",
    "plot_zscore_bar",
    "plot_zscore_heatmap",
    "plot_graph_with_motif",
    "plot_top_motif",
    "run_pipeline",
]
