# SuperAgent — Standalone All-English Presentation Draft & 6–7 Minute Verbatim Script

- **Course & Venue**: AIAA 3102 Course Project Roadshow & Deep Q&A Defense · HKUST (Guangzhou)
- **Target Duration**: **6 min 31 sec – 6 min 45 sec** (calibrated strictly inside the **6–7 minute** window at **912 spoken English words** / **135–140 wpm**)
- **Primary Focus**: **Evolutionary Tree Search Architecture & Contract-First Project Design** (multi-branch orthogonal seeding, UCT ↔ diversified elite exploration–exploitation, local chain state & anti-race plateaus, 7 specialized operators, complementarity value-density scheduling, AST-scoped patching, contrastive RRF memory, and `stratified_halves` out-of-sample ensembling; operational token/cost hygiene kept concisely in Backup `B3`)
- **Deck File**: `SuperAgent_Presentation_EN.pptx` (`16:9`, `10.0 in × 5.625 in`, official HKUST(GZ) template; synchronized word-for-word in Slide Speaker Notes)
- **PDF Companion**: `SuperAgent_Presentation_EN.pdf`
- **Deck Structure**: **12 Core Slides** (`Slides 1–12`, timed) + **1 Backup Divider** (`Slide 13`) + **4 Deep Q&A Backup Slides** (`Slides 14–17 / B1–B4`, un-timed)
- **Custom Vector Graphics**: 7 W3C SVG flowcharts/architecture diagrams (`assets/svg/*.svg`) + 4 300-DPI Matplotlib empirical plots (`assets/plot_*.png`)

---

## 1. Master Pacing & Timing Budget (Slides 1–12)

| Slide | Section | Slide Title | Visual Asset | Time Window | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Opening | *SuperAgent: Contract-Verified Autonomous ML via Adaptive Evolutionary Tree Search* | HKUST Title Layout | `0:00 – 0:25` | **25s** |
| **2** | Motivation | *Why Naive LLM Agents Fail in Autonomous ML Engineering* | `svg_01_failure_modes.svg` | `0:25 – 1:00` | **35s** |
| **3** | Architecture | *System Architecture: Strict Adapter–Core 1.1 Separation* | `svg_02_architecture.svg` | `1:00 – 1:35` | **35s** |
| **4** | Search I | *Search Engine I: Multi-Branch Tree, UCT–Elite & Chain Control* | `svg_03_search_tree.svg` | `1:35 – 2:15` | **40s** |
| **5** | Search II | *Search Engine II: 7 Operators & Value-Density Admission* | `svg_04_operators_adaptive.svg` | `2:15 – 2:55` | **40s** |
| **6** | Memory & Recall | *Hybrid Knowledge Base: Qwen3 + CamelCase BM25 + RRF* | `svg_05_hybrid_memory.svg` | `2:55 – 3:30` | **35s** |
| **7** | Finalization | *Trustworthy Finalization: Stratified Two-Half Gate & Prob Sidecars* | `svg_06_validated_ensemble.svg` | `3:30 – 4:05` | **35s** |
| **8** | Evaluation I | *Evaluation I: Multi-Modality Vision & Multi-Branch Search Health* | `plot_02_multimodal_search.png` | `4:05 – 4:35` | **30s** |
| **9** | Evaluation II | *Evaluation II: Eliminating the Timeout Spiral & Metric Drift* | `plot_03_timeout_fix.png` | `4:35 – 5:05` | **30s** |
| **10** | Head-to-Head | *Head-to-Head: SuperAgent vs. MLEvolve (4× RTX 4090, 1h)* | Table + `plot_04_head2head.png` | `5:05 – 5:40` | **35s** |
| **11** | Hardening | *Task-Agnostic Generalization & Concurrency Hardening* | `svg_07_generalization.svg` | `5:40 – 6:10` | **30s** |
| **12** | Conclusion | *Conclusion: Three Design Principles for Autonomous ML Search* | 3-Principle Summary Cards | `6:10 – 6:30` | **20s** |
| **Total** | **Core Deck** | **12 Core Slides (+ B1–B4 Backup Defense)** | **7 SVGs + 4 Plots + Tables** | **`0:00 – 6:30`** | **6m 30s** |

---

## 2. Slide-by-Slide Visual Draft & Word-for-Word English Script

---

### Slide 1 · Title Slide (`0:00 – 0:25` | 25s)
- **On-Slide Draft**:
  - **Title**: *SuperAgent: Contract-Verified Autonomous ML via Adaptive Evolutionary Tree Search*
  - **Subtitle**: *Multi-Branch UCT–Elite Search · 7 Evolutionary Operators · Hybrid RRF Memory · Stratified OOS Ensembling*
  - **Footer Metadata**: *AIAA 3102 Course Project Roadshow · HKUST (Guangzhou) · Evaluated on 8× NVIDIA RTX 4090 Server · 541 Passing Automated Contract & Unit Tests*
- **Delivery Cue**: Warm, confident opening; pause slightly after "SuperAgent," and emphasize *"evolutionary tree search inside a strict zero-leakage harness"*.
- **Verbatim Script**:
  > "Good morning, professors and classmates. Today we're excited to present SuperAgent, our autonomous machine learning system for AIAA 3102. Rather than relying on trial-and-error code generation, we treat AutoML as an evolutionary tree search inside a strict zero-leakage harness—exploring diverse model families, learning from past runs, and verifying every ensemble out-of-sample."

---

### Slide 2 · Why Naive LLM Agents Fail in Autonomous ML Engineering (`0:25 – 1:00` | 35s)
- **On-Slide Draft**:
  - **SVG Diagram (`svg_01_failure_modes.svg`)**: Four side-by-side columns contrasting *Problem* (red) vs. *SuperAgent Remedy* (teal):
    1. *Data & Validation Leakage* $\rightarrow$ Physical `private/` firewall + inner-CV only rule + AST/LLM leak audit.
    2. *Greedy Search & Branch Collapse* $\rightarrow$ Orthogonal family seeding (`t=0`) + UCT ↔ Elite 7-op search tree + contrastive `(+1/-1)` RRF memory.
    3. *Script Timeout Death Spiral* $\rightarrow$ Decoupled priority vs. slot-scaled kill ceiling (`31 -> 1` timeout).
    4. *Fragile Metric & Blind Fusion* $\rightarrow$ Task-hint + `1e-4` exact-hit metric ID + `stratified_halves` OOS gate + prob-sidecar & `log_loss` tie-break.
  - **Bottom Callout**: *Core Design Thesis: Principled Evolutionary Search Inside Deterministic Contracts*.
- **Delivery Cue**: Sweep left-to-right across the four columns, landing firmly on how each failure maps directly to a SuperAgent safeguard.
- **Verbatim Script**:
  > "Why is that structure necessary? In multi-hour ML runs, naive LLM agents break down in four predictable ways. First, subtle data or validation leakage—like scaling before splitting, or tuning blend weights on validation labels—creates inflated scores that collapse at test time. Second, greedy search gets trapped in local optima while parallel workers repeat the same dead ends. Third, timeouts tuned for quick baselines accidentally kill deeper five-fold GBDTs and neural nets. Finally, hard-label voting throws away model confidence and overfits. Each SuperAgent module tackles one failure head-on."

---

### Slide 3 · System Architecture: Strict Adapter–Core 1.1 Separation (`1:00 – 1:35` | 35s)
- **On-Slide Draft**:
  - **SVG Flowchart (`svg_02_architecture.svg`)**: Three-zone closed-loop pipeline:
    - **Zone 1 (Input Adapter Layer)**: `prepare` & `check_isolation` build sanitized `task/` while locking `private/test_labels.csv` behind a red **STRICT LABEL FIREWALL**.
    - **Zone 2 (SuperAgent Core 1.1 Search Engine, 4 GPUs · 4 Workers)**: `Profiler & Method Catalog` $\rightarrow$ `AdaptiveController` + `Hybrid ExperienceStore` $\rightarrow$ `7 Operators & AST-Scoped Patching` $\rightarrow$ `Risk-Gated Review & GPU Executor` $\rightarrow$ `Independent Assessment & Validated Final Builder (final.py)`.
    - **Zone 3 (Verified Delivery)**: `core11.py Offline Audit` (zero Core execution, symlink lineage, byte-exact `combine.py` replay) $\rightarrow$ `Final Evaluation & Report`.
  - **Bottom Bar**: Three boundaries (*Physical Label Firewall*, *Task-Agnostic Core*, *Offline Replay Gate*).
- **Delivery Cue**: Trace the red dashed firewall underneath Core 1.1 to show that `private/test_labels.csv` never enters the search workspace.
- **Verbatim Script**:
  > "Starting with leakage prevention, we split the system into two decoupled layers connected strictly through our Core 1.1 filesystem contract. On the left, the Course Adapter locks private test labels outside the agent workspace behind a physical firewall—the agent never even sees them. In the center, Core 1.1 coordinates four parallel GPU workers that profile the data, launch distinct model families, evolve a shared search tree with hybrid memory, and assemble a gated final bundle. On the right, `core11.py` replays `combine.py` offline, byte-for-byte, before any grading begins."

---

### Slide 4 · Search Engine I: Multi-Branch Tree, UCT–Elite & Chain Control (`1:35 – 2:15` | 40s)
- **On-Slide Draft**:
  - **SVG Diagram (`svg_03_search_tree.svg`)**:
    - *Left Panel (`selection.py`)*: `Root` $\rightarrow$ 3 concurrent orthogonal drafts (`Branch 1: TabICL / GBDT / ConvNeXt`, `Branch 2: Emb-MLP / FT-Trans / DINOv2`, `Branch 3: CatBoost+ET / EffNetV2`). Dual paths: `tree_walk` UCT (`c(t): 1.414 -> 0.50`, `max_refinements=3`, `max_repairs=1`) vs. `elite_pick` diversified pool (`early_k=5 (3/branch) -> late_k=3 (2/branch)`, `w_i ~ 1/rank_i`, `elite_max_children=10`, `chain_soft_switch`, thread-local `Dispatch.via_elite`), controlled by `explore_weight(p)` decay bar (`1.0` for `p<0.5` $\rightarrow$ `0.2` for `p>=0.7`).
    - *Right Panel (`chain.py`, `scheduling.py`)*: Smooth `step_reward = min(1.5, 0.75 + g / max(|s_prev|, 0.05))`, per-node local `no_gain_count` (`max_no_gain=3`), repair bounds (`return=3, max=20`), `forced_return` (`p>=0.4` every `3` steps; `p>=0.8` `50%`) with `recent_window=4` exemption, and `_after_best_finished` anti-race plateau window (`branch=3`, `elite=6`).
- **Delivery Cue**: Point first to the explore–exploit decay bar on the left, then sweep right to the concurrency safeguards.
- **Verbatim Script**:
  > "Now let's look inside the search engine. From the start, we seed orthogonal model families in parallel—like TabICL, LightGBM, and CatBoost for tabular data, or ConvNeXt and DINOv2 for vision. Early on, we explore with UCT, decaying its constant from 1.414 down to 0.5. Past the halfway mark, `explore_weight` smoothly shifts toward `elite_pick`, sampling a branch-diverse top pool weighted by rank with thread-local state. Along each branch, smooth step rewards up to 1.5 favor real breakthroughs, while per-node stall counters, periodic root returns, and finish-time plateau filters prevent worker race conditions."

---

### Slide 5 · Search Engine II: 7 Operators & Value-Density Admission (`2:15 – 2:55` | 40s)
- **On-Slide Draft**:
  - **SVG Diagram (`svg_04_operators_adaptive.svg`)**:
    - *Left Panel (`operators.py`, `scheduling.py`)*: Tiered showcase of all 7 operators: Tier 1 (`1. draft`, `2. refine` with AST import-aware `SEARCH/REPLACE`, `3. repair` incl. 1 tier-2 failed-draft retry), Tier 2 (`4. retrospect` 10-step trajectory + escalation, `5. promote` zero-LLM Successive-Halving checkpoint continuation verified by `optimizer_steps` & hash), Tier 3 (`6. borrow` cross-branch grafting past `p>=0.5`, `7. synthesis` global-plateau multi-branch fusion in `[0.5, 0.83]`).
    - *Right Panel (`control/adaptive.py`)*: Complementarity-guided value-density admission `priority = merit * p_ok / max(need, 1)` where `merit` adds `min(0.5, comp)` (`comp = mean((best != y_val) & (node == y_val))`), multi-slot `rank_proposals()` (`slot=k`), autonomous `target=None`, and decoupled `slot_share` kill ceiling (`max(2.5*q0.8(ok), 0.75*slot_share)`).
- **Delivery Cue**: Highlight `promote` (zero LLM cost), `borrow` and `synthesis` (plateau crossover), and `complementarity` rewarding error diversity over greedy top-1 chasing.
- **Verbatim Script**:
  > "Across these branches, seven specialized operators drive the search. Beyond initial drafts, `refine` makes targeted AST-scoped edits that handle new imports, `repair` diagnoses crashes from tracebacks, and `promote` extends saved checkpoints via Successive Halving with zero LLM calls. When a branch or the search plateaus past 50 percent progress, `retrospect` audits the last ten steps while `borrow` and `synthesis` recombine complementary branches. To decide what runs next, `AdaptiveController` ranks proposals by value density—balancing rank, novelty, and how well a model fixes the current best's mistakes—under a slot-scaled timeout floor."

---

### Slide 6 · Hybrid Knowledge Base: Qwen3 + CamelCase BM25 + RRF (`2:55 – 3:30` | 35s)
- **On-Slide Draft**:
  - **SVG Diagram (`svg_05_hybrid_memory.svg`)**: Dual local CPU `Qwen/Qwen3-Embedding-0.6B` (1024-d, `cos >= 0.35`, zero GPU contention) + `CamelCase + CJK` Okapi BM25 (`k1=1.5, b=0.75`, e.g., `StratifiedKFold -> [stratifiedkfold, stratified, kfold]`) fused via `Weighted RRF (k=60)` into the Evidence-Guided Planner (`Top-2 +1` gains & `Top-2 -1` prioritizing direct failed children).
  - **Bottom Callout**: *Complementary Recall* + *Contrastive Anti-Looping*.
- **Delivery Cue**: Emphasize zero GPU contention from local CPU `Qwen3-0.6B`, CamelCase symbol splitting, and contrastive `+1 / -1` recall.
- **Verbatim Script**:
  > "To keep parallel workers from repeating the same mistakes, we built a hybrid `ExperienceStore` that logs every run's plan, score change, and traceback. Before planning the next step, Reciprocal Rank Fusion blends two signals: semantic embeddings from a local CPU `Qwen3-0.6B` model—freeing up the GPUs—and a CamelCase- and CJK-aware BM25 index that catches exact symbols like `StratifiedKFold` or CUDA errors. Injecting the top two wins alongside the top two failures, prioritizing direct failed children, steers the planner away from dead ends."

---

### Slide 7 · Trustworthy Finalization: Stratified Two-Half Gate & Prob Sidecars (`3:30 – 4:05` | 35s)
- **On-Slide Draft**:
  - **SVG Pipeline (`svg_06_validated_ensemble.svg`)**: `1. Top-6 Diverse Candidates` (`top_per_branch=5`, `val/test_probabilities.csv`) $\rightarrow$ `2. Metric & Candidate Plans` (`1e-4 exact_hits`, odd-pair greedy + `prob_argmax`) $\rightarrow$ `3. stratified_halves OOS Gate` (Half A $\leftrightarrow$ Half B double win + `log_loss > 0.001` tie-break) $\rightarrow$ `4. Verified Primary final/` (Intel Image `95.33%` or zero-harm single fallback + `core11.py` replay).
- **Delivery Cue**: Emphasize the two-way `stratified_halves` check—if an ensemble can't win out-of-sample on both halves, we safely fall back to the best single model.
- **Verbatim Script**:
  > "Once search finishes, naive agents either blindly average their top scripts or overfit blend weights directly on validation labels. SuperAgent enforces a strict `stratified_halves` Two-Half gate: weights fit on stratified Half A must beat the best single model on held-out Half B, and vice versa. And because hard labels throw away confidence, our probability sidecars, odd-pair greedy search, and log-loss tie-breaker unlock calibrated soft blending—still with byte-exact offline replay."

---

### Slide 8 · Evaluation I: Multi-Modality Vision & Multi-Branch Search Health (`4:05 – 4:35` | 30s)
- **On-Slide Draft**:
  - **300-DPI Plot (`plot_02_multimodal_search.png`)**:
    - *Left Panel*: Per-class test F1 across 3,000 sealed Intel Image test photos (`99.68%` forest .. `91.24%` glacier), showing **Validated `top6` Ensemble `95.33%`** beating **Best Single `95.13%`**.
    - *Right Panel*: Search execution node outcomes across 5 branches (`78/120` = `65.0%` $\rightarrow$ `72/85` = **`84.7%` on Vision** $\rightarrow$ `49/57` = **`86.0%` on Airline v1** with `11` zero-LLM `promote`s).
- **Delivery Cue**: Point to the `95.33%` vs. `95.13%` gain on the left and the healthy `84.7%`–`86.0%` valid-node rate on the right.
- **Verbatim Script**:
  > "Now let's see how this performs in practice. Without changing a single line of core code, we evaluated SuperAgent on both vision and tabular benchmarks using RTX 4090 GPUs. On the six-class Intel Image benchmark—with 3,000 sealed test photos—SuperAgent explored 85 nodes across five diverse model families, including ConvNeXt, Swin, and EVA-02. Our Two-Half gate accepted a six-model ensemble that reached 95.33 percent private test accuracy, beating the 95.13 percent best single model, while maintaining an 84.7 percent valid execution rate."

---

### Slide 9 · Evaluation II: Eliminating the Timeout Spiral & Metric Drift (`4:35 – 5:05` | 30s)
- **On-Slide Draft**:
  - **300-DPI Plot (`plot_03_timeout_fix.png`)**:
    - *Left Panel*: Node success rate rising `65.0% -> 86.0%` while timeout-killed nodes drop `31 -> 1`.
    - *Right Panel*: Closing the `0.37%` mixed OOF vs. `val.csv` gap to **`0.00%`** (`92.94% == 92.94%`).
- **Delivery Cue**: Frame this as closed-loop systems debugging: spot the bottleneck in telemetry, fix the root cause, and verify the jump quantitatively.
- **Verbatim Script**:
  > "Moving to the 80,000-row Airline tabular benchmark, our telemetry uncovered two subtle system bugs. First, quick 15-second baselines dragged our quantile timeout down to 25 seconds, prematurely killing 31 deeper GBDT and neural runs. Decoupling expected runtime from our slot-scaled kill ceiling cut timeout kills from 31 down to 1, boosting node success from 65 to 86 percent. Second, exact metric disambiguation eliminated a 0.37 percent OOF mismatch, locking in exact 92.94 percent `val.csv` alignment."

---

### Slide 10 · Head-to-Head: SuperAgent vs. MLEvolve (`4× RTX 4090`, 1h) (`5:05 – 5:40` | 35s)
- **On-Slide Draft**:
  - **Left Table**: Controlled 1h comparison (`80k dev / 20k test`, `4× RTX 4090`, `16 CPUs`, `deepseek-flash`):
    - Nodes: **57 (`49 ok`)** vs. `30 (27 ok)`
    - Zero-LLM Promotes: **11** vs. `0`
    - Search Throughput: **1.9× (`53.6m`)** vs. `1.0× (60.0m)`
    - Primary Test Acc: **92.175%** vs. `92.170%`
    - Best Single Test Acc: **92.245% (`rank_06`)** vs. `92.235% (top3)`
    - Ensemble Governance: **Stratified OOS Gate** vs. `Unvalidated Prob Avg`
  - **Right Plot (`plot_04_head2head.png`)**: `1.9×` node throughput + sealed test accuracy edge on both Primary and Best Single.
- **Delivery Cue**: Anchor on **search throughput and model quality** (`57 vs. 30` nodes = `1.9×`, `11` zero-LLM promotes, `92.245%` best single); mention cost/token efficiency only if asked (Backup `B3`).
- **Verbatim Script**:
  > "Next, we ran a controlled one-hour head-to-head against MLEvolve on the exact same 80,000-row Airline split, four RTX 4090 GPUs, and DeepSeek backend. Driven by value-density scheduling and 11 zero-LLM checkpoint promotions, SuperAgent explored 57 nodes versus MLEvolve's 30—1.9 times the search throughput. On the 20,000-row sealed test set, SuperAgent edged ahead on primary accuracy at 92.175 percent versus 92.170 percent, and reached 92.245 percent on our best single model—while MLEvolve's soft blending directly inspired our probability sidecar upgrade."

---

### Slide 11 · Task-Agnostic Generalization & Concurrency Hardening (`5:40 – 6:10` | 30s)
- **On-Slide Draft**:
  - **SVG Cards (`svg_07_generalization.svg`)**:
    1. *Concurrency-Safe Tree Search* (per-node `no_gain_count`, `_after_best_finished` anti-race filter, multi-slot `rank_proposals`, `Dispatch.via_elite`).
    2. *AST Patching & Smart Recall* (AST import-aware `SEARCH/REPLACE`, autonomous `target`, CamelCase+CJK BM25, direct failed-child `-1` priority).
    3. *Stratified OOS & Sidecar Gate* (`stratified_halves`, `val/test_probabilities.csv`, `+2` pair greedy, anti-val-snooping checks).
    4. *Orthogonal Seed & Slot Scaling* (`assign_draft_hypotheses()` at `t=0`, decoupled `slot_share`, zero-LLM `promote`, `541` tests).
  - **Bottom Verification Box**: `410 Core unit tests + 131 Adapter contract tests = 541 passed, 0 failed`.
- **Delivery Cue**: Emphasize zero dataset hardcoding and point to the bottom `541`-test verification bar.
- **Verbatim Script**:
  > "Every insight from those runs went straight back into four task-agnostic upgrades—zero dataset hardcoding. We hardened multi-worker search with per-node stall state and plateau filters. We upgraded code edits and recall with import-aware AST patching and CamelCase BM25 memory. We added `stratified_halves` probability sidecars and anti-snooping audits, alongside orthogonal seeding and slot-scaled timeouts. Across both Core and Adapter, all 541 automated unit and contract tests pass with zero failures."

---

### Slide 12 · Conclusion: Three Design Principles for Autonomous ML Search (`6:10 – 6:30` | 20s)
- **On-Slide Draft**:
  - **Three Principle Cards**:
    - *1 · Contract-First Zero-Leakage Isolation* (`private/` firewall + `val.csv` exact scoring + offline `core11.py` replay).
    - *2 · Principled Evolutionary Tree Search* (orthogonal seeding + UCT ↔ Elite 7-op search, complementarity value-density + contrastive RRF memory).
    - *3 · Stratified Out-of-Sample Ensembling* (exact-hit metric ID + `stratified_halves` OOS gate + prob sidecars & `log_loss` tie-break).
  - **Headline Summary**: `95.33%` Intel Image · `1.9×` search throughput vs. MLEvolve · `92.245%` best Airline single · `541/541` tests passing.
- **Delivery Cue**: Pause briefly before the headline numbers, finish crisply around `6:30`, and invite Q&A.
- **Verbatim Script**:
  > "To wrap up, SuperAgent brings together three design principles: strict contract isolation against leakage, adaptive evolutionary tree search powered by complementarity and hybrid memory, and stratified out-of-sample ensembling. Together, they deliver 95.33 percent vision accuracy, 1.9 times MLEvolve's search throughput, and 541 passing tests. Thank you for listening—we'd love to take your questions!"

---

## 3. Quick-Jump Guide for Q&A Backup Slides (`Slides 14–17 / B1–B4`)

| Professor / Judge Question Topic | Jump To | Key Evidence to Point Out on Screen |
| :--- | :--- | :--- |
| *"How do you guarantee zero test leakage, prevent validation snooping, and audit `combine.py`?"* | **Slide 14 (`B1`)** | 4-layer defense: `private/` isolated outside `task_dir`, pre-run AST/risk audit, `val.csv` re-scoring + jump check, and zero-Core `course/back/core11.py` byte-exact CSV replay. |
| *"Show the exact math for UCT, `step_reward`, complementarity merit, `slot_share`, RRF, and two-half CV."* | **Slide 15 (`B2`)** | Exact formulas: `UCB` (`c: 1.414->0.5`), smooth `step_reward ∈ [0.75, 1.5]`, `merit` + `comp`, `slot_share` ceiling, `0.5/(60+r_kw)+0.5/(60+r_vec)`, and `stratified_halves` + `log_loss > 0.001`. |
| *"How do you keep thinking tokens under control and avoid runaway parallel API spend?"* | **Slide 16 (`B3`)** | Engineering appendix: CoT kept for `planner/draft`, stripped on mechanical `patch_coder` (`-68.8%` CoT), plus optional SQLite pre-send ledger & `warn_fraction` soft wind-down. |
| *"How does SuperAgent's search engine compare to AIDE / MLEvolve, and what hardening did you integrate?"* | **Slide 17 (`B4`)** | 6-row comparison matrix + full hardening summary (`per-node no_gain_count`, `_after_best_finished`, multi-slot `rank_proposals`, AST import patching, CamelCase+CJK BM25, `stratified_halves`, `541` passing tests). |
