# Your Null Model Is Your Claim: Signed Motifs and Coherent Reinforcement in Transformer Circuits

## TL;DR

In [Part 1](https://open2interp.substack.com), I reported that feedforward loops (FFLs) are the dominant structural motif in 99 attribution graphs from Claude 3 Haiku. But attribution graphs are layered DAGs — their architecture generates FFLs "for free." How much of the signal is real?

I ran the same 99 graphs through four progressively stricter null models. **FFLs survive all four.** They are the only motif class that does. But the progression from Z = +107 (Erdos-Renyi) to Z = +18 (layer-pair configuration model) shows that ~80% of the original signal was architectural. The residual is genuine learned wiring.

Then I asked a question the unsigned census can't: are the model's FFLs *coherent* (both paths reinforce) or *incoherent* (paths oppose)? This is the most important functional distinction in Alon's motif framework — and it's invisible to standard analysis.

Across 99 graphs and 1,000 randomizations per graph:
- **Coherent FFLs are enriched** (Z = +12.6) — the model builds more convergent reinforcing topologies than hub structure predicts.
- **Incoherent FFLs are depleted** (Z = −5.4) — but only when you shuffle signs. The topology is unremarkable; the model's sign placement specifically prevents incoherence.
- **Cross-chain inhibition is depleted** (Z = −8.5) — parallel paths don't carry opposing signs.
- **Cross-chain cooperation is enriched** (Z = +3.0) — parallel paths preferentially agree.

These statistics describe aggregate structure. To see what they look like in practice, I traced signed motif cascades through individual circuits. A 22-step cascade through Haiku's refusal circuit — "How do I make a bomb?" — shows coherent FFLs chaining from generic "Assistant" features through increasingly specific refusal representations, with every single motif instance coherent. A rhyming circuit builds the word "rabbit" through parallel phonological and lexical streams that converge via coherent amplification. Across 9 task categories, the main cascade is entirely coherent in 7 of 9 — with the two exceptions (code, arithmetic) being exactly the tasks where output competition between discrete candidates requires dampening.

The design principle: **coherent reinforcement through convergent paths.** The model builds multi-path circuits, assigns reinforcing signs, and avoids adversarial configurations — a pattern that echoes Alon's finding that coherent FFLs dominate gene regulatory networks, now recovered in a completely different computational substrate.

---

## 1. The Problem: What Does "Enriched" Actually Mean?

In the original analysis, I followed Alon and Milo et al.'s recipe: count motifs in the real network, count motifs in degree-preserving randomizations (the configuration model), compute Z-scores. For gene regulatory networks, this is the right question. The degree distribution is a basic structural property, and enrichment relative to it reflects genuine wiring logic.

But attribution graphs have additional structure: **they are layered DAGs.** Every node has a layer assignment (embeddings at the bottom, transcoder features in the middle, logits at the top), and edges only flow forward. This layered structure is baked into the transformer architecture — it isn't learned.

Consider three nodes at layers 5, 10, and 15. If layer-5 connects to both layer-10 and layer-15, and layer-10 connects to layer-15, that's an FFL. It arose because all three layer pairs happened to have edges — not because the model built a feedforward loop for a computational purpose. The configuration model doesn't know about layers, so it can't distinguish architectural FFLs from learned ones.

This isn't a hypothetical concern. It's straightforward to construct layered DAGs where the configuration model produces **zero variance** — 1,000 rewirings produce identical motif counts every time. When layer structure is tight, the degree sequence leaves no freedom for the triad census to vary. The configuration model becomes uninformative. This forced me to think carefully about what a null model for attribution graphs needs to control.

## 2. Four Null Models, Four Questions

I implemented four null models forming a hierarchy from most permissive to most strict. Each controls for additional structure, and the enrichment that survives each successive control tells you something different.

[FIGURE: fig_unrolled_universal_enrichment.png — triple heatmap showing mean Z-scores, % enriched, and % depleted across all 5 null models × 8 signed motifs]

### 2.1 Configuration Model (Degree-Preserving Rewiring)

**Preserves:** Exact in-degree and out-degree of every node.
**Randomizes:** Which specific nodes connect to which. Edges can be rewired "backward" — the null graphs aren't even DAGs.
**Enrichment means:** "This motif appears more than the degree distribution alone predicts."

For 99 Haiku graphs: FFLs at Z = +25.9 (99/99 enriched), chains at Z = +20.0 (96/99). But this conflates learned wiring with architectural forward-flow.

### 2.2 Erdos-Renyi (Global Random)

**Preserves:** Node count and edge count only.
**Randomizes:** Everything.
**Enrichment means:** "More than a random graph of the same density." The weakest possible claim.

FFLs at Z = +107 (even larger, because ER doesn't preserve degree distribution). Fan-in at Z = +82 — an artifact of the layered architecture funneling features toward logit nodes.

### 2.3 Layer-Pair ER (Architecture-Preserving Random)

**Preserves:** Edge count between each (source_layer, target_layer) pair — the DAG "skeleton."
**Randomizes:** Degree distribution within each layer pair. Hub nodes are destroyed.
**Enrichment means:** "Given the layer architecture, does *specific connectivity* create more motifs than random wiring within layer pairs?"

**FFLs survive** (Z = +93.6, 99/99 enriched). Fan-in collapses from Z = +82 to +5.2 — confirming it was architectural. Chains flip to **strongly depleted** (−50.6) — the layer architecture predicts *more* chains than observed, because the model "uses" its edge budget for skip connections (FFLs) rather than sequential relay.

Key insight: **the model preferentially builds skip connections over sequential chains, given the same architectural budget.**

### 2.4 Layer-Pair Configuration Model (Architecture + Hub Preserving)

**Preserves:** Edge count per layer pair AND per-node degree within each pair. Hub structure is fully preserved.
**Randomizes:** Only which specific targets each source connects to, within degree constraints.
**Enrichment means:** "Given architecture AND hub structure, does the *specific wiring* create motifs beyond what both can explain?" The strictest test.

**FFLs still survive** (Z = +17.5, 96/99 enriched). Weaker — some FFL enrichment was explained by hub structure. But the residual is real. Everything else is depleted (Z = −17.5), reflecting a zero-sum redistribution forced by the degree constraint.

### 2.5 Summary: What Survives

| Motif | Config | ER | LP-ER | LP-Config |
|-------|:---:|:---:|:---:|:---:|
| **FFL (030T)** | **+26** ✓ | **+107** ✓ | **+94** ✓ | **+18** ✓ |
| Fan-in (021U) | −1 | +82 | +5 | −18 |
| Fan-out (021D) | −11 | +15 | −18 | −18 |
| Chain (021C) | +20 | −13 | −51 | −18 |

FFLs are the only motif enriched under all four null models. The progression from Z = +107 to +18 quantifies how much is architectural (~80%) versus learned (~20%). But the learned residual is large, consistent, and present in 96/99 individual graphs.

## 3. Unrolling Triads: From Topology to Function

While developing the null model hierarchy, Joshua Batson raised a point that cuts deeper than null model choice:

> I think of Alon's graph as really making most sense in a recurrent network (like a gene network) where there is a time component. But here things flow forward. So vanilla mutual inhibition is impossible; you'd need something like two chains which inhibit each other. Did you look into any "unrolled" analogues of the classic regulatory motifs?

This is the conceptual version of what the null model analysis showed statistically. Alon's motifs were discovered in gene regulatory networks, which are *recurrent* — a gene can regulate another gene that regulates it back. The triad census was designed for networks where all 16 patterns can occur. In a layered DAG, half are structurally impossible, and the rest are constrained by forward flow.

More importantly, Alon's most functionally significant distinction — **coherent versus incoherent feedforward loops** — is invisible to the unsigned census. A coherent FFL (both paths reinforce the target) and an incoherent FFL (paths oppose each other) have identical topology but opposite computational functions. In gene regulation, the coherent FFL acts as a persistence detector; the incoherent FFL acts as a pulse generator. The standard triad census treats them as the same pattern.

To address both concerns, I extended the analysis to signed motifs, incorporating edge polarity (excitatory vs. inhibitory attribution) and "unrolling" the motif vocabulary to include DAG-native patterns like cross-chain interactions.

### 3.1 The Signed Motif Vocabulary

For each triad instance, I classified edges by sign and split base motifs into signed variants:

| Signed Motif | Structure | Alon Analogue |
|---|---|---|
| **Coherent FFL** | A→B→C, A→C; paths agree in sign | Persistence detector |
| **Incoherent FFL** | A→B→C, A→C; paths disagree | Pulse generator |
| **FF Amplification** | A→B→C, A→C; both excitatory | Signal amplification |
| **FF Dampening** | A→B→C, A→C; mixed with net dampening | Signal attenuation |
| **Cross-chain together** | Parallel chains converging with same sign | Cooperative convergence |
| **Cross-chain inhibition** | Parallel chains converging with opposing signs | Mutual inhibition (unrolled) |

For FFL sign classification, I follow Alon: compare the sign of the direct path (A→C) against the sign of the indirect path (sign(A→B) × sign(B→C)). Coherent = same, incoherent = opposite.

### 3.2 Two Null Models for Signs

To separate topological structure from sign structure, I ran two variants of the strictest null:

**LPC-shuf (signs shuffled):** Rewire topology within each layer pair preserving degrees, then randomly assign signs from the graph's global excitatory/inhibitory ratio. This jointly tests whether the model builds more of a signed motif than random topology + random signs would produce.

**LPC-sign (signs preserved):** Same topological randomization, but signs stay attached to their positions. This tests purely whether the *topology* is non-random — the sign classification just partitions the count.

The comparison between these two nulls is the key decomposition. If a signed motif is enriched under LPC-sign but neutral under LPC-shuf, topology is learned but sign placement is random. If depleted under LPC-shuf but neutral under LPC-sign, the depletion is entirely a sign effect — the topology is unremarkable, but the real graph's signs are arranged to prevent that pattern.

## 4. Results: 99 Graphs, 1,000 Rewirings

### 4.1 The Full Table

| Motif | Config | ER | LP-ER | LPC-shuf | LPC-sign |
|-------|:---:|:---:|:---:|:---:|:---:|
| **Coherent FFL** | +100 | +158 | +9.8 | **+4.8** | **+12.6** |
| Incoherent FFL | +99 | +47 | −3.6 | **−5.4** | −0.1 |
| FF Amplification | +100 | +55 | +1.0 | −0.1 | +0.2 |
| FF Dampening | +100 | +50 | +1.8 | +0.6 | −0.7 |
| Cross-chain inhibition | +99 | +125 | −4.4 | **−8.5** | −1.0 |
| Cross-chain together | +99 | +280 | +6.4 | **+3.0** | −0.1 |
| SelfLoop+ | +100 | +31 | −0.5 | −0.5 | +0.0 |
| SelfLoop− | −10 | +21 | +0.5 | +0.5 | +0.0 |

Mean Z-scores across 99 graphs. Bold = significant (|Z| > 2) in >50% of individual graphs.

[FIGURE: fig_unrolled_sign_effect.png — paired bar chart showing LPC-shuf vs LPC-sign for all 8 motifs, with TOPO and SIGN annotations]

### 4.2 The Topology Story (LPC-sign column)

Coherent FFL at Z = +12.6 is the dominant topological signal. The model builds substantially more FFL wiring than degree-preserving randomization within layer pairs would produce. This specific wiring — which sources connect to which targets — creates convergent multi-path structures beyond what hub degrees explain. It's the strongest result in the table and is significant in >50% of individual graphs.

Everything else is near zero. The topology of cross-chain patterns, dampening patterns, and amplification patterns is unremarkable given the degree sequence.

### 4.3 The Sign Story (LPC-shuf vs LPC-sign gap)

This is where the new information lives. The gap between LPC-shuf and LPC-sign isolates the effect of sign placement, independent of topology.

**Incoherent FFL:** LPC-sign = −0.1 (completely neutral topology) but LPC-shuf = −5.4 (strongly depleted). The model builds exactly as many incoherent-FFL-shaped topologies as expected — there's no topological avoidance. But its signs are arranged so that far fewer of those topologies end up incoherent than random sign assignment would produce.

**Cross-chain inhibition:** LPC-sign = −1.0 (marginal) but LPC-shuf = −8.5 (strongly depleted). The second-largest absolute Z-score under LPC-shuf. The model's sign placement actively prevents parallel paths from carrying opposing signs.

**Cross-chain together:** LPC-sign = −0.1 (neutral) but LPC-shuf = +3.0 (enriched). The positive complement: the model doesn't just avoid inhibitory cross-chains, it builds cooperative ones.

### 4.4 The Decomposition

| Finding | Topological (LPC-sign) | Sign effect (gap) | Interpretation |
|---|---|---|---|
| FFL topology is non-random | **+12.6** | −7.8 (dilution) | Model builds extra FFL wiring |
| Signs on FFLs are coherent | −0.1 | **−5.3** (avoidance) | Sign placement prevents incoherence |
| Parallel paths cooperate | −0.1 | **+3.1** (preference) | Signs favor agreement |
| Parallel paths don't compete | −1.0 | **−7.5** (avoidance) | Signs prevent opposition |

Two independent design principles, cleanly separated:

**Topology:** The model builds more FFL wiring than hub structure predicts — a genuine connectivity choice.

**Signs:** Independently of topology, signs are arranged so that multi-path circuits reinforce rather than compete. This affects FFLs (coherent > incoherent), cross-chains (cooperative > inhibitory), and is completely invisible to unsigned analysis.

## 5. From Statistics to Circuits: The Motif Cascade

Statistics describe aggregate structure. To see what coherent reinforcement looks like in practice, I traced signed motif cascades through individual circuits — finding connected chains of motif instances from embedding to output, weighted by attribution magnitude.

### 5.1 A Safety Refusal: "How do I make a bomb?"

[FIGURE: fig_unrolled_cascade_safety_graph.png]

The cascade anchors on the "Assistant" token and builds a 22-step, fully connected chain from L1 to L13. The feature labels read like a progressive refinement:

"Assistant" (L1) → "Assistant about to do a task" (L3) → "Assistant on potential refusal" (L4) → "assistant when its identity..." (L5) → "about to say a topic is..." (L6) → "Assistant on prompt which i..." (L9) → "responses on difficult t..." (L11) → "responses, mostly refusals" (L12) → "Assistant in meta" (L13)

Every single motif instance is coherent — coherent FFLs and FF amplification, no incoherent patterns, no dampening. The circuit doesn't build "refuse" by suppressing "comply." It progressively amplifies the correct response through reinforcing multi-path convergence. Total path weight: 268.1, the highest of any category.

### 5.2 A Rhyming Couplet: "grab it" → "rabbit"

[FIGURE: fig_unrolled_cascade_creative_graph.png]

The creative cascade reveals phonological constraint satisfaction happening through motif chains. It grounds on "grab" (L1-L3), then splits into two parallel streams: phonetic structure ("say something that ends in it" at L5, L10, L15) and lexical candidates ("habit" at L8-L10, "say ab" at L10, "say a word with a b in it" at L14). The streams converge at L15-L16 where "say rabbit" emerges. Again, 22 steps, all coherent, second-highest total weight (235.4).

### 5.3 Multihop: "The capital of the state containing Dallas"

[FIGURE: fig_unrolled_cascade_multihop_graph.png]

The Dallas graph shows a structural gap — the cascade splits into an early segment (embedding → "state" features at L1-L3) and a late segment (L7-L17, "Austin/Texas" → "say a capital" → "say Austin"). The gap at L3-L7 means the intermediate computation (Dallas → Texas) doesn't form 3-node motif patterns at this weight threshold, possibly occurring through attention head operations that the transcoder features don't decompose. This is consistent with multihop's outlier status in the aggregate statistics — its circuits have less dense motif coverage.

### 5.4 Code: Where Dampening Lives

[FIGURE: fig_unrolled_cascade_code_graph.png]

`a = "Craig"; assert a[0] == ""` → the model needs to output "C." The cascade is FF-amplification-dominated (13 steps, late-starting at L7) with visible inhibitory edges and FF dampening near the logit layer. Competing outputs "S" and "r" are dampened while "C" wins via the coherent pathway. This is the first circuit where output competition appears explicitly in the motif cascade — and it's in exactly the task category you'd predict: symbolic manipulation where discrete candidates must be adjudicated.

The arithmetic graph ("7 14 21 28 35 " → "42") shows the same pattern: a compact late-layer cascade with dampening at the output.

### 5.5 Cross-Category Comparison

[FIGURE: fig_unrolled_cascade_comparison.png — path length, composition, and weight across 9 categories]

| Category | Steps | Gap? | Dominant Type | Weight |
|---|---|---|---|---|
| factual_recall | 23 | no | Coherent FFL | 143.6 |
| creative | 22 | yes | Coherent FFL | 235.4 |
| safety | 22 | no | Coherent FFL | 268.1 |
| arithmetic | 18 | no | Coherent FFL | 143.4 |
| uncategorized | 18 | no | Coherent FFL | 75.6 |
| code | 13 | no | FF Amplification | 78.8 |
| multihop | 10 | yes | FF Amplification | 142.3 |
| reasoning | 8 | yes | Coherent FFL | 116.7 |
| multilingual | 7 | no | Coherent FFL | 113.5 |

Three regimes emerge:

**Deep cascades** (factual recall, creative, safety: 22-23 steps): Knowledge and language tasks with processing chains spanning most of the network. Dominated by coherent FFLs. Highest weights.

**Medium cascades** (arithmetic, uncategorized: 18 steps): Substantial but more compact.

**Short/gapped cascades** (code, multihop, reasoning, multilingual: 7-13 steps): Either the computation is spatially concentrated (code doing symbolic work in late layers only) or the 3-node motif vocabulary doesn't capture the circuit structure well (multihop and reasoning, which both have gaps suggesting computation through mechanisms the signed triad census misses).

## 6. Discussion

### What this means for the original finding

The FFL enrichment from Part 1 is valid and robust — it survives the strictest null model. But the null model comparison adds nuance: ~80% of the signal is architectural (hub structure within layer pairs), and the remaining ~20% reflects genuine learned wiring. The model doesn't just have high-degree nodes that happen to create FFLs — it wires those hubs' outputs into specific convergent patterns.

### What signed motifs reveal

The unsigned census captured topology but missed function. The most consistent signal in these data isn't topological FFL enrichment — it's **sign coherence**. The depletion of incoherent FFLs (Z = −5.4) and cross-chain inhibition (Z = −8.5) under LPC-shuf, combined with their neutral topology under LPC-sign, demonstrates that the model's edge signs are a second, independent layer of learned structure. Multi-path circuits reinforce rather than compete, and this is invisible to any unsigned analysis.

This recovers Alon's central insight from gene regulation — that the distinction between coherent and incoherent FFLs is more functionally important than the FFL topology itself — in a completely different computational substrate.

### The cascade connection

The aggregate statistics and individual cascades tell consistent stories. The statistical enrichment of coherent FFLs maps onto circuits where progressive refinement cascades build representations through reinforcing multi-path convergence. The statistical depletion of dampening and inhibition maps onto a model that selects correct outputs primarily through amplification of the right answer rather than suppression of wrong ones — with the notable exception of symbolic tasks (code, arithmetic) where discrete disambiguation requires explicit competition.

The safety cascade is particularly striking: a 22-step chain of coherent reinforcement from "Assistant" to "refusal," with the model's entire response strategy readable from the feature labels. This suggests that signed motif cascades may be a useful tool for circuit-level interpretability — not just for characterizing aggregate structure, but for tracing specific computations through the network.

### Why these motifs? Noise filtering in superposition

Uri Alon has argued that the motifs dominating a network reveal what that network is optimized to do. Gene transcription networks in *E. coli* and neural networks in *C. elegans* share the same motif profile — coherent FFLs enriched, the same anti-motifs — despite operating at completely different physical scales with different substrates. His explanation: both evolved to transduce information between noisy inputs and noisy components. The motif profile is a fingerprint of the optimization pressure, not the medium.

Transformer circuits face an analogous problem. MLP layers operate in superposition — far more features are represented than there are dimensions, so features share dimensions and interfere with each other. The transcoder features in this analysis are decompositions of this superposed signal, but the underlying activations are a high-dimensional space where the "right" features coexist with many irrelevant ones. From the model's perspective at each layer, the problem is: route the relevant signal forward while irrelevant features create interference. This is structurally parallel to Alon's framing — process information from noisy inputs with noisy components.

The coherent FFL may serve the same function in both systems. In *E. coli*, the direct path A→C provides immediate signal while the indirect path A→B→C provides delayed, filtered confirmation. The coherent FFL only fully activates the target when both paths agree, filtering out transient noise — a spurious input activates the direct path but not the slower indirect one, so the target doesn't fire. In the transformer, the direct path (a skip connection from an earlier feature to a later one) provides a "shortcut" signal, while the indirect path (through an intermediate feature) provides a processed, refined version. When both paths carry the same sign (coherent), the model has convergent evidence that the signal is genuine rather than a superposition artifact.

In this framing, sign coherence isn't just a structural regularity — it's a functional requirement for reliable signal routing through a noisy medium. An incoherent FFL, where the direct and indirect paths disagree, would amplify uncertainty rather than resolve it. The depletion of incoherent FFLs reflects the same constraint that limits biological circuits to the few designs robust to component noise. Cross-chain inhibition (parallel paths disagreeing) would similarly represent contradictory conclusions from independent processing streams — exactly what noise-filtering should prevent.

This gives a testable prediction: coherent FFL enrichment should be stronger in layers or positions where superposition is more severe (more features competing for fewer dimensions), because the noise-filtering function is more necessary there. The fact that our cascades show dense FFL coverage in middle-to-late layers and sparser coverage near embeddings — where token representations are relatively clean — is directionally consistent, though not a definitive test.

One important caveat: Alon's networks evolve under selection pressure with genuinely stochastic noise from molecular concentrations. Transformers are trained by gradient descent, and superposition interference is deterministic, not stochastic. The analogy is suggestive — both systems face a signal-extraction-from-interference problem and converge on similar circuit designs — but the underlying mechanisms differ. Whether gradient descent discovers coherent FFLs for the same functional reasons that evolution discovers them, or through some other optimization dynamic, remains an open question.

### Limitations

**Descriptive, not causal.** We know the model builds coherent multi-path circuits, but not why. Is sign coherence a byproduct of gradient descent? Is it functionally necessary? Intervention experiments (breaking sign coherence and measuring performance degradation) would address this.

**Effect sizes.** Z-scores measure statistical significance relative to null variance, not computational importance. The absolute excess FFL count may be small relative to the total. The fraction of attribution weight flowing through FFL edges versus non-FFL edges would better measure computational significance.

**Single model.** All results are from Claude 3 Haiku with Anthropic's transcoders. Whether the coherent-FFL design principle holds across architectures, scales, and training procedures is an open question.

**Triad limitations.** The 3-node signed motif vocabulary may be too small to capture the computational structures of compositional reasoning, as suggested by the short cascades and structural gaps in multihop and reasoning categories. Larger motifs (4-5 nodes) or path-based analysis may be needed.

### Next Steps

**Cross-model comparison.** The signed motif profile as a computational fingerprint could be used to compare models across training procedures, scales, and architectures. Does RLHF change the coherence profile? Does scale affect it? Does distillation preserve or destroy it?

**Position-aware nulls.** The current nulls treat all nodes within a layer as interchangeable. A position × layer null could control for additional structure from token-position connectivity patterns.

**Causal validation.** Interventions that break sign coherence in specific FFLs (flipping the sign of the direct edge in a coherent FFL to make it incoherent) and measuring downstream effects on model output would test whether coherence is functionally necessary or merely a structural regularity.

**Larger motifs.** Extending to 4-node signed patterns could capture the "broadcast-then-compete" structures that Josh suggested, and might explain the computational content of tasks (like multihop) where triads leave structural gaps.

---

## Appendix: Software and Reproducibility

- **Attribution graphs:** 99 graphs from [Neuronpedia](https://www.neuronpedia.org/), generated by Anthropic's circuit-tracer methodology on Claude 3 Haiku
- **Motif analysis pipeline:** [circuit-motifs](https://github.com/mkenney2/circuit-motifs) (open source)
- **Null model implementations:**
  - Configuration: igraph `rewire()` with `n_edges × 10` swap attempts
  - Erdos-Renyi: igraph `Erdos_Renyi(n, m)`
  - Layer-pair ER: Custom — random bipartite graphs per layer pair, same edge budget
  - Layer-pair config: Custom — degree-preserving bipartite swaps per layer pair
  - LPC-shuf / LPC-sign: Layer-pair config + sign shuffling or preservation
- **Signed census:** VF2 subgraph isomorphism with sign and layer constraints
- **Cascade analysis:** Heaviest-path extraction through motif instance overlap graph
- **Graph library:** igraph
- **Statistical tests:** scipy (Mann-Whitney U, Kruskal-Wallis, Spearman correlation)

---

*This is Part 2 of a series on network motif analysis of transformer circuits. [Part 1](https://open2interp.substack.com) introduced the circuit-motifs pipeline and reported FFL enrichment across 99 graphs.*
