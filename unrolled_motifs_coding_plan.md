# Coding Plan: Unrolled Regulatory Motifs for `circuit-motifs`

## Motivation

The standard triad census (16 isomorphism classes of 3-node directed subgraphs) correctly identifies FFLs and chains as enriched, and correctly reports that all mutual-edge and cyclic motifs are depleted/absent. But the *reason* those motifs are absent is architectural (transformers are feedforward), not functional. The question is: **are the computational functions served by recurrent motifs (mutual inhibition, feedback, toggle switching) achieved through feedforward-compatible wiring patterns spread across layers?**

This extension adds a new analysis mode to `circuit-motifs` that searches for "unrolled" analogues of Alon's classic regulatory motifs — patterns that respect the layer-ordering constraint of transformer computation graphs.

---

## Integration with Existing Codebase

### Current `circuit-motifs` architecture (from repo):

```
src/
├── graph_loader.py         # Parse circuit-tracer JSON → igraph DiGraph
├── motif_census.py         # Triad census + VF2 instance finding
├── null_model.py           # Degree-preserving randomization + Z-scores
├── comparison.py           # Cross-task SP vectors, statistical tests
├── visualization.py        # Neuronpedia-style drawing, heatmaps
├── pipeline.py             # Batch processing
└── neuronpedia_client.py   # Fetch graphs from Neuronpedia / S3
```

### What we reuse directly:
- **`graph_loader.py`** — loads attribution graphs, already parses node metadata (including layer info) and edge weights/signs
- **`null_model.py`** — the degree-preserving rewiring engine (needs a wrapper to add layer-ordering constraint)
- **`visualization.py`** — the Neuronpedia-style layout (already positions nodes by layer on y-axis)
- **`pipeline.py`** — batch processing pattern (extend for unrolled analysis)
- **`comparison.py`** — SP normalization, cosine similarity, statistical tests (reuse as-is for unrolled motif profiles)

### New files to add:

```
src/
├── ... (existing files unchanged) ...
├── unrolled_motifs.py       # NEW: Motif template definitions + matching
├── unrolled_null_model.py   # NEW: Layer-preserving null model
├── unrolled_census.py       # NEW: Enumeration pipeline (orchestrates the above)
└── unrolled_visualization.py # NEW: Layered motif diagrams
```

Plus a new notebook: `notebooks/unrolled_motif_analysis.ipynb`

---

## Phase 1: Extract Layer Info from Existing Graphs

**Goal**: Verify we can reliably get layer indices for every node.

**File**: No new file — inspect `graph_loader.py` output.

### Tasks:
1. Load a few attribution graphs and inspect node attributes. The circuit-tracer JSON should have layer info per feature (e.g., `"layer": 5` or encoded in the feature ID like `L5/feature_1234`).
2. Write a utility function `get_layer_index(node) -> int` that extracts layer from whatever format is used.
3. Verify that all edges in the loaded graphs go forward in layer index (source layer < target layer). The blog post says the directed 3-cycle count is literally zero, so this should hold, but confirm programmatically.
4. Compute and log: how many distinct layers per graph, distribution of nodes per layer, distribution of edge "layer gaps" (target_layer - source_layer).

**Output**: A helper function and a quick stats summary confirming layer structure.

```python
# Pseudocode
def get_layer_index(graph, node_id) -> int:
    """Extract layer index from node attributes."""
    # Adapt to whatever format graph_loader.py uses
    ...

def validate_layer_ordering(graph) -> bool:
    """Confirm all edges go forward in layer index."""
    for edge in graph.es:
        src_layer = get_layer_index(graph, edge.source)
        tgt_layer = get_layer_index(graph, edge.target)
        assert src_layer < tgt_layer, f"Backward edge: {edge.source}(L{src_layer}) -> {edge.target}(L{tgt_layer})"
    return True

def layer_gap_distribution(graph) -> dict:
    """Histogram of (target_layer - source_layer) for all edges."""
    ...
```

---

## Phase 2: Define the Unrolled Motif Catalog

**Goal**: Formalize each classic recurrent motif as a layer-respecting feedforward template.

**File**: `src/unrolled_motifs.py`

### The catalog:

Each motif is defined as a small directed graph template with:
- **Nodes**: labeled with a `chain_id` (which logical "stream" the node belongs to) and a `relative_layer` (ordering constraint, not absolute layer)
- **Edges**: labeled with `sign` (+1 or -1) and `type` ("intra-chain" or "cross-chain")
- **Constraints**: layer ordering rules (strict inequality along chains)

```python
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import igraph as ig

@dataclass
class UnrolledMotifTemplate:
    """A feedforward-compatible analogue of a classic regulatory motif."""
    name: str                           # e.g., "cross_chain_inhibition"
    classic_analogue: str               # e.g., "mutual_inhibition"
    description: str
    nodes: List[dict]                   # [{"id": "A1", "chain": 0, "relative_order": 0}, ...]
    edges: List[dict]                   # [{"src": "A1", "tgt": "B2", "sign": -1}, ...]
    min_layer_gap: int = 1              # minimum layer distance for each edge
    max_layer_gap: int = 5              # maximum layer distance (search window)
    
    def to_template_graph(self) -> ig.Graph:
        """Convert to igraph for VF2 matching."""
        ...
    
    def num_nodes(self) -> int:
        return len(self.nodes)
```

### Motif definitions (7 templates):

#### 1. Cross-Chain Inhibition (unrolled mutual inhibition)
- Classic: A ⊣ B, B ⊣ A
- Unrolled (4 nodes):
  ```
  A_early ──(+)──> A_late
  B_early ──(+)──> B_late
  A_early ──(−)──> B_late    (cross-inhibition)
  B_early ──(−)──> A_late    (cross-inhibition)
  ```
- Constraint: layer(A_early) < layer(A_late), layer(B_early) < layer(B_late), all cross edges also go forward

#### 2. Feedforward Damping (unrolled negative feedback)
- Classic: A → B ⊣ A
- Unrolled (3 nodes):
  ```
  A_early ──(+)──> B_mid ──(−)──> A_late
  ```
- Constraint: layer(A_early) < layer(B_mid) < layer(A_late)
- Note: "A_early" and "A_late" = same feature type at different layers, OR same logical role (see Phase 2b)

#### 3. Feedforward Amplification (unrolled positive feedback)
- Classic: A → B → A
- Unrolled (3 nodes):
  ```
  A_early ──(+)──> B_mid ──(+)──> A_late
  ```
- Same structure as damping but all-excitatory

#### 4. Residual Self-Loop (unrolled autoregulation)
- Classic: A → A (positive) or A ⊣ A (negative)
- Unrolled (2 nodes):
  ```
  A_early ──(+/−)──> A_late
  ```
- Simplest case: same feature appearing at two layers with a direct edge
- Split into positive (self-reinforcement) and negative (self-suppression) variants

#### 5. Cross-Chain Toggle (unrolled bistable switch)
- Classic: A ⊣ B, B ⊣ A, with upstream bias
- Unrolled (5 nodes):
  ```
  Bias ──(+)──> A_mid ──(−)──> B_late
  Bias ──(+)──> B_mid ──(−)──> A_late
  A_mid ──(+)──> A_late   (optional self-reinforcement)
  B_mid ──(+)──> B_late   (optional self-reinforcement)
  ```
- This is the most speculative — may not exist

#### 6. Coherent FFL (already in census, include for completeness)
- 030T with all-positive or sign-coherent edges
- Already detected by existing `motif_census.py`, but re-annotate with sign info

#### 7. Incoherent FFL
- 030T with sign-incoherent edges (one path +, other path −)
- The "output competition" pattern you found in the Dallas circuit (FFL #3 with the inhibitory shortcut)

### Phase 2b: Node Identity Matching

This is the hardest design decision. What counts as "the same node at different layers"?

**Option A: Same SAE feature index at different layers**
- Most conservative. Only matches if the exact same feature ID appears at two layers.
- Problem: features are layer-specific in most transcoder architectures, so this may never match.

**Option B: Semantic similarity of feature labels**
- Use the feature descriptions from Neuronpedia (e.g., "Dallas-related" at L5 and "Dallas/Texas" at L10).
- Requires embedding feature labels and computing cosine similarity above a threshold.
- More flexible but noisier.

**Option C: Structural role matching (no identity requirement)**
- Don't require A_early and A_late to be "the same" — just require the edge pattern and sign pattern to match.
- This is the most permissive and probably the right starting point.
- The motif structure IS the finding; whether the nodes are semantically related is a separate analysis.

**Recommendation**: Start with **Option C** (pure structural matching), then add Option B as an enrichment filter in Phase 4.

---

## Phase 3: Subgraph Matching Engine

**Goal**: Enumerate instances of each unrolled motif template in an attribution graph.

**File**: `src/unrolled_census.py`

### Approach:

For 3-4 node motifs, we can use igraph's VF2 subgraph isomorphism (same approach as `motif_census.py`'s `find_motif_instances`), but with added constraints:

```python
def find_unrolled_instances(
    graph: ig.Graph,
    template: UnrolledMotifTemplate,
    weight_threshold: float = 0.0,
    max_layer_gap: int = 5,
) -> List[UnrolledMotifInstance]:
    """
    Find all instances of an unrolled motif in the attribution graph.
    
    Uses VF2 with custom feasibility checks:
    1. Layer ordering: edges must go forward in layer index
    2. Layer gap: no edge spans more than max_layer_gap layers
    3. Sign matching: edge signs must match template
    4. Weight threshold: edges must exceed minimum |weight|
    """
    template_graph = template.to_template_graph()
    
    instances = []
    
    # Use igraph's get_subisomorphisms_vf2 with node/edge compatibility
    mappings = graph.get_subisomorphisms_vf2(
        template_graph,
        node_compat_fn=_node_layer_compatible,
        edge_compat_fn=_edge_sign_compatible,
    )
    
    for mapping in mappings:
        instance = UnrolledMotifInstance(
            template=template,
            node_ids=mapping,
            layers=[get_layer_index(graph, n) for n in mapping],
            edge_weights=_extract_edge_weights(graph, mapping, template),
            total_weight=_compute_instance_weight(graph, mapping, template),
        )
        
        # Post-filter: check layer gap constraint
        if instance.max_layer_gap <= max_layer_gap:
            instances.append(instance)
    
    # Deduplicate (same nodes, different mapping order)
    instances = _deduplicate(instances)
    
    # Sort by total weight
    instances.sort(key=lambda x: x.total_weight, reverse=True)
    
    return instances
```

### Key implementation detail — sign-aware matching:

```python
def _edge_sign_compatible(graph, template_graph, graph_edge_id, template_edge_id):
    """Check that edge signs match between graph and template."""
    graph_weight = graph.es[graph_edge_id]["weight"]
    template_sign = template_graph.es[template_edge_id]["sign"]
    
    if template_sign == +1:
        return graph_weight > 0
    elif template_sign == -1:
        return graph_weight < 0
    else:  # template_sign == 0 means "any sign"
        return True
```

### Handling the edge sign data:

The blog post mentions that most FFL edges are excitatory but FFL #3 in the Dallas circuit had an inhibitory shortcut edge. This means the graph loader already preserves sign info in edge weights. **Confirm**: are weights signed (positive/negative) or unsigned with a separate sign attribute? This determines how `_edge_sign_compatible` works.

### Complexity estimate:

For 4-node motifs on graphs with ~100-400 nodes: VF2 is fast enough. The layer-ordering constraint actually helps prune the search space dramatically since most node pairs won't satisfy the layer constraint. Expect seconds per graph, not minutes.

---

## Phase 4: Layer-Preserving Null Model

**Goal**: A randomized baseline that preserves layer ordering, so enrichment scores reflect genuine structural preferences, not just the constraint that edges go forward.

**File**: `src/unrolled_null_model.py`

### Key insight:

The existing `null_model.py` uses `igraph.rewire()` for degree-preserving randomization. But for unrolled motifs, we need a null that ALSO preserves:
1. Layer ordering (no backward edges after rewiring)
2. Edge sign distribution
3. Layer-gap distribution (ideally)

### Algorithm: Layer-Constrained Degree-Preserving Rewiring

```python
def layer_preserving_rewire(graph: ig.Graph, n_swaps: int = 1000, seed: int = None) -> ig.Graph:
    """
    Degree-preserving random rewiring that maintains:
    1. In/out degree of each node
    2. Forward layer ordering of all edges
    3. Edge sign distribution (positive/negative)
    
    Uses the standard switching algorithm but rejects swaps that
    would create backward edges.
    """
    g = graph.copy()
    rng = np.random.default_rng(seed)
    
    edges = list(g.es)
    n_accepted = 0
    n_attempted = 0
    
    while n_accepted < n_swaps and n_attempted < n_swaps * 10:
        # Pick two random edges with the same sign
        e1, e2 = rng.choice(len(edges), size=2, replace=False)
        
        if g.es[e1]["sign"] != g.es[e2]["sign"]:
            n_attempted += 1
            continue
        
        s1, t1 = g.es[e1].source, g.es[e1].target
        s2, t2 = g.es[e2].source, g.es[e2].target
        
        # Propose swap: (s1→t1, s2→t2) becomes (s1→t2, s2→t1)
        l_s1 = get_layer_index(g, s1)
        l_t2 = get_layer_index(g, t2)
        l_s2 = get_layer_index(g, s2)
        l_t1 = get_layer_index(g, t1)
        
        # Check layer ordering is preserved
        if l_s1 >= l_t2 or l_s2 >= l_t1:
            n_attempted += 1
            continue
        
        # Check no multi-edges created
        if g.are_connected(s1, t2) or g.are_connected(s2, t1):
            n_attempted += 1
            continue
        
        # Check no self-loops
        if s1 == t2 or s2 == t1:
            n_attempted += 1
            continue
        
        # Accept swap
        g.delete_edges([e1, e2])  # careful with index invalidation
        g.add_edge(s1, t2, **edge_attrs_from(g.es[e1]))
        g.add_edge(s2, t1, **edge_attrs_from(g.es[e2]))
        
        n_accepted += 1
        n_attempted += 1
    
    return g
```

### Null ensemble + Z-scores:

```python
def compute_unrolled_zscores(
    graph: ig.Graph,
    templates: List[UnrolledMotifTemplate],
    n_random: int = 1000,
    **kwargs,
) -> dict:
    """
    For each unrolled motif template:
    1. Count instances in the real graph
    2. Count instances in n_random layer-preserving rewirings
    3. Compute Z = (real - mean_null) / std_null
    """
    real_counts = {}
    for tmpl in templates:
        instances = find_unrolled_instances(graph, tmpl, **kwargs)
        real_counts[tmpl.name] = len(instances)
    
    null_counts = {tmpl.name: [] for tmpl in templates}
    
    for i in range(n_random):
        g_null = layer_preserving_rewire(graph, n_swaps=graph.ecount())
        for tmpl in templates:
            null_instances = find_unrolled_instances(g_null, tmpl, **kwargs)
            null_counts[tmpl.name].append(len(null_instances))
    
    z_scores = {}
    for tmpl in templates:
        obs = real_counts[tmpl.name]
        null = np.array(null_counts[tmpl.name])
        z_scores[tmpl.name] = (obs - null.mean()) / (null.std() + 1e-10)
    
    return z_scores, real_counts, null_counts
```

### Performance concern:

Running VF2 on 1000 null graphs × 7 motif templates could be slow. Mitigations:
- Parallelize with `multiprocessing.Pool` (each null graph is independent)
- Start with n_random=100 for development, scale to 1000 for publication
- Cache null counts per graph

---

## Phase 5: Integration with Existing Pipeline

**File**: Extend `src/pipeline.py`

### Add an `--unrolled` flag:

```python
# In pipeline.py, add:
def run_unrolled_analysis(graph, templates, n_random=1000, max_layer_gap=5):
    """Run unrolled motif analysis on a single graph."""
    z_scores, real_counts, null_counts = compute_unrolled_zscores(
        graph, templates, n_random=n_random, max_layer_gap=max_layer_gap
    )
    
    # Get top instances for each motif type
    top_instances = {}
    for tmpl in templates:
        instances = find_unrolled_instances(graph, tmpl, max_layer_gap=max_layer_gap)
        top_instances[tmpl.name] = instances[:10]  # top 10 by weight
    
    return {
        "z_scores": z_scores,
        "real_counts": real_counts,
        "null_summary": {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in null_counts.items()},
        "top_instances": top_instances,
    }
```

### Output format:

Extend `analysis_summary.json` with an `"unrolled_motifs"` key per graph, containing z-scores and top instances. This keeps backward compatibility.

---

## Phase 6: Visualization

**File**: `src/unrolled_visualization.py`

### 1. Layered motif instance diagram

Draw individual motif instances with:
- Y-axis = layer index (reuse existing visualization.py layout)
- Nodes colored by chain membership
- Edges colored by sign (green = excitatory, red = inhibitory)
- Edge width proportional to |weight|

### 2. Unrolled motif spectrum

Bar chart of Z-scores across all 7 unrolled motif types, analogous to the existing triad census Z-score plot. Overlay the classic triad census results for comparison.

### 3. Cross-motif co-occurrence

Heatmap: for each pair of unrolled motif types, how often do their instances share nodes? This reveals whether e.g. cross-chain inhibition tends to co-locate with FFLs (suggesting they work together) or is spatially separated.

---

## Phase 7: Analysis Notebook

**File**: `notebooks/unrolled_motif_analysis.ipynb`

### Sections:

1. **Layer structure validation** — confirm all edges go forward, plot layer-gap distribution
2. **Unrolled motif census** — run across all 99 graphs, generate Z-scores
3. **Key question: do the unrolled recurrent analogues exist?**
   - Is cross-chain inhibition enriched? (Would mean the model implements mutual suppression between competing feature streams)
   - Is feedforward damping enriched? (Would mean the model implements self-regulation)
   - How do counts compare to the dominant FFL/chain motifs?
4. **Case study: Dallas circuit revisited** — You already found the inhibitory FFL #3 that suppresses the generic "capital" response. Is this part of a larger cross-chain inhibition motif when you look at the 4-node pattern?
5. **Task-type differences** — Do certain task categories use more inhibitory unrolled motifs? (Hypothesis: safety/refusal circuits might show more cross-chain inhibition)
6. **SP profiles with unrolled motifs** — Append the 7 unrolled Z-scores to the existing 16 triad Z-scores for a 23-dimensional profile. Redo the cosine similarity and clustering analysis.

---

## Execution Order

| Step | What | Estimated Effort | Depends On |
|------|------|-----------------|------------|
| 1 | Phase 1: Layer info extraction + validation | 2 hours | Nothing (exploratory) |
| 2 | Phase 2: `unrolled_motifs.py` — template definitions | 3 hours | Step 1 (need to know layer format) |
| 3 | Phase 3: `unrolled_census.py` — VF2 matching with constraints | 4 hours | Step 2 |
| 4 | Unit tests for matching (known small graphs) | 2 hours | Step 3 |
| 5 | Phase 4: `unrolled_null_model.py` — layer-preserving rewiring | 3 hours | Step 1 |
| 6 | Combine: Z-score computation pipeline | 2 hours | Steps 3 + 5 |
| 7 | Run on 1 graph (Dallas circuit) as sanity check | 1 hour | Step 6 |
| 8 | Phase 5: Pipeline integration + batch run on 99 graphs | 2 hours | Step 7 |
| 9 | Phase 6: Visualization | 3 hours | Step 8 |
| 10 | Phase 7: Analysis notebook + writeup | 4 hours | Step 9 |

**Total: ~26 hours of focused work**

---

## Key Risks and Mitigations

1. **Node identity problem (Phase 2b)**: If we require "same feature at different layers" and transcoders are per-layer, we may find zero matches for damping/amplification motifs. **Mitigation**: Start with Option C (pure structural matching, no identity requirement). The motif is the edge pattern, not the node labels.

2. **Null model acceptance rate**: Layer-preserving rewiring may have low acceptance rates for dense graphs, making the null model slow to mix. **Mitigation**: Monitor acceptance rate; if < 5%, switch to a sampling-based null (generate random DAGs with matched degree sequence and layer structure) rather than edge-switching.

3. **Multiple testing**: 7 new motif types × 99 graphs = 693 tests. **Mitigation**: Use Bonferroni or BH correction, same as the existing analysis.

4. **Interpretability of 4+ node motifs**: Cross-chain inhibition has 4 nodes, toggle has 5. VF2 is still fast at this size, but the number of instances could explode combinatorially. **Mitigation**: Weight-threshold edges before matching (only consider edges with |weight| > some percentile), and cap instance enumeration at top-K by weight.

5. **Sign data availability**: The plan assumes edge weights are signed. If they're stored as unsigned magnitudes, we need to go back to the circuit-tracer output to recover signs. **Mitigation**: Check this first in Phase 1.

---

## Expected Outcomes (Hypotheses)

- **Cross-chain inhibition**: Likely enriched in output-competition layers (final 3-4 layers). The Dallas circuit already shows competing pathways ("say Austin" vs "say a capital") — this motif would formalize that.
- **Feedforward damping**: May appear in safety/refusal circuits where the model needs to self-suppress.
- **Feedforward amplification**: Likely enriched and may partially overlap with chained FFLs.
- **Residual self-loop**: Almost certainly enriched (the residual stream trivially creates these). The interesting question is whether self-loops are more positive or negative.
- **Cross-chain toggle**: Speculative. If found, it would suggest the model implements something like attention-based routing or winner-take-all competition.
