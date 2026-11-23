"""Build AIAA 3102 SuperAgent English Roadshow Presentation on official HKUST(GZ) template.

Structure:
- Slides 1–12: Core 6–7 minute English presentation (~6m30s, ~905 spoken words at ~139 wpm)
  deeply focused on Search Architecture & Core Project Design.
- Slide 13: Backup Section Header
- Slides 14–17 (B1–B4): Deep Q&A Defense Backup Slides (Anti-Leakage & core11 Replay,
  Search & Ensembling Mathematical Formulations, Engineering Hygiene / Cost Telemetry Appendix,
  SuperAgent vs MLEvolve Matrix & Integrated Hardening).

Output: SuperAgent_Presentation_EN.pptx
"""
import copy
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
A = ROOT / "assets"
TEMPLATE = ROOT / "src" / "HKUST_Template.pptx"
OUT = ROOT / "SuperAgent_Presentation_EN.pptx"

NAVY = RGBColor(0x11, 0x2B, 0x5C)
BLUE = RGBColor(0x1D, 0x4E, 0xD8)
TEAL = RGBColor(0x0F, 0x76, 0x6E)
GOLD = RGBColor(0x92, 0x40, 0x0E)
RED = RGBColor(0xB9, 0x1C, 0x1C)
PURPLE = RGBColor(0x6D, 0x28, 0xD9)
DARK = RGBColor(0x1E, 0x29, 0x3B)
GREY = RGBColor(0x47, 0x55, 0x69)
LIGHT = RGBColor(0xF1, 0xF5, 0xF9)
PALEB = RGBColor(0xEF, 0xF6, 0xFF)
PALET = RGBColor(0xF0, 0xFD, 0xFA)
PALEO = RGBColor(0xFF, 0xFB, 0xEB)
PALEP = RGBColor(0xFA, 0xF5, 0xFF)
BORDER = RGBColor(0xCB, 0xD5, 0xE1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Arial"

prs = Presentation(TEMPLATE)
SW, SH = prs.slide_width, prs.slide_height
tpl_slides = list(prs.slides)
TPL_CONTENT = tpl_slides[2]
LAYOUT = TPL_CONTENT.slide_layout
SECTION_LAYOUT = [l for l in prs.slide_layouts if l.name == "SECTION_HEADER"][0]


def _strip_placeholders(slide):
    for sh in list(slide.placeholders):
        sh._element.getparent().remove(sh._element)


def new_slide(title, title_size=21.0):
    s = prs.slides.add_slide(LAYOUT)
    _strip_placeholders(s)
    for sh in TPL_CONTENT.shapes:
        if sh.is_placeholder and sh.placeholder_format.idx == 1:
            continue
        s.shapes._spTree.append(copy.deepcopy(sh._element))
    t = s.shapes.title
    bp = t.text_frame._txBody.find(qn("a:bodyPr"))
    for c in list(bp):
        bp.remove(c)
    t.text_frame.text = title
    r = t.text_frame.paragraphs[0].runs[0]
    r.font.size = Pt(title_size)
    r.font.color.rgb = NAVY
    r.font.bold = True
    r.font.name = FONT
    return s


def tb(slide, x, y, w, h, paras, size=14.5, color=DARK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=4.5, font=FONT):
    box_ = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box_.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(spacing)
        runs = [(p, {})] if isinstance(p, str) else p
        for text, st in runs:
            r = para.add_run()
            r.text = text
            f = r.font
            f.name = font
            f.size = Pt(st.get("size", size))
            f.bold = st.get("bold", False)
            f.italic = st.get("italic", False)
            f.color.rgb = st.get("color", color)
    return box_


def card_box(slide, x, y, w, h, title, body_lines, fill=LIGHT, line=BORDER, title_color=NAVY, title_size=13.2, body_size=11.8, lw=1.4, space_after=2.2):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line
    sh.line.width = Pt(lw)
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_right = Inches(0.11)
    tf.margin_top = Inches(0.07)
    tf.margin_bottom = Inches(0.05)

    p0 = tf.paragraphs[0]
    p0.alignment = PP_ALIGN.LEFT
    p0.space_after = Pt(space_after + 0.8)
    r0 = p0.add_run()
    r0.text = title
    r0.font.name = FONT
    r0.font.size = Pt(title_size)
    r0.font.bold = True
    r0.font.color.rgb = title_color

    for ln in body_lines:
        p = tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(space_after)
        if isinstance(ln, list):
            for txt, st in ln:
                r = p.add_run()
                r.text = txt
                r.font.name = FONT
                r.font.size = Pt(st.get("size", body_size))
                r.font.bold = st.get("bold", False)
                r.font.color.rgb = st.get("color", DARK)
        else:
            r = p.add_run()
            r.text = ln
            r.font.name = FONT
            r.font.size = Pt(body_size)
            r.font.color.rgb = DARK
    return sh


def pic(slide, path, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def table(slide, rows, x, y, w, col_w, size=11.0, header_fill=NAVY, row_h=0.31, bold_first_col=True, hl_rows=()):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    t = gt.table
    tblPr = gt._element.graphic.graphicData.tbl.tblPr
    style = tblPr.find(qn("a:tableStyleId"))
    if style is not None:
        style.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i in range(nr):
        t.rows[i].height = Inches(row_h)
        for j in range(nc):
            cell = t.cell(i, j)
            cell.text = ""
            cell.margin_left = cell.margin_right = Inches(0.05)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            r = p.add_run()
            r.text = str(rows[i][j])
            r.font.name = FONT
            r.font.size = Pt(size)
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = header_fill
                r.font.bold = True
                r.font.color.rgb = WHITE
            else:
                cell.fill.fore_color.rgb = PALET if i in hl_rows else (WHITE if i % 2 else LIGHT)
                r.font.color.rgb = DARK
                r.font.bold = (j == 0 and bold_first_col) or (i in hl_rows)
    return gt


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text.strip()


def footer(slide, text, y=5.07, w=9.1, size=9.5):
    tb(slide, 0.35, y, w, 0.25, [[(text, {"color": GREY, "size": size})]], spacing=0)


# ==============================================================================
# SLIDE 1 · Title Slide (0:00–0:25 | 25s)
# ==============================================================================
s1 = tpl_slides[0]
for sh in s1.placeholders:
    idx = sh.placeholder_format.idx
    if idx == 0:
        tf = sh.text_frame
        tf.text = "SuperAgent: Contract-Verified Autonomous ML via Adaptive Evolutionary Tree Search"
        r = tf.paragraphs[0].runs[0]
        r.font.size = Pt(26.5)
        r.font.bold = True
        r.font.color.rgb = NAVY
        r.font.name = FONT
    elif idx == 1:
        tf = sh.text_frame
        bp = tf._txBody.find(qn("a:bodyPr"))
        for c in list(bp):
            bp.remove(c)
        for p in tf.paragraphs[1:]:
            p._p.getparent().remove(p._p)
        for r_ in list(tf.paragraphs[0].runs):
            r_._r.getparent().remove(r_._r)
        lines = [
            ("Multi-Branch UCT–Elite Search · 7 Evolutionary Operators · Hybrid RRF Memory · Stratified OOS Ensembling", 14.0, BLUE, True),
            ("AIAA 3102 Course Project Roadshow · HKUST (Guangzhou)", 17.5, DARK, True),
            ("Evaluated on 8× NVIDIA RTX 4090 Server · 541 Passing Automated Contract & Unit Tests", 14.0, GREY, False),
        ]
        for i, (t_, sz, col, b_) in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            p.space_after = Pt(7)
            r = p.add_run()
            r.text = t_
            r.font.size = Pt(sz)
            r.font.color.rgb = col
            r.font.bold = b_
            r.font.name = FONT

notes(s1, """
[Slide 1 | 0:00–0:25 | 25s]
Good morning, professors and classmates. Today we present SuperAgent, our autonomous machine learning engineering system for AIAA 3102. Instead of treating AutoML as blind prompt-and-retry code generation, we formulate it as principled evolutionary tree search inside a strict zero-leakage harness—exploring diverse model families, learning from contrastive hybrid memory, and verifying ensembles out-of-sample.
""")


# ==============================================================================
# SLIDE 2 · Motivation: 4 Failure Modes of Naive LLM Agents (0:25–1:00 | 35s)
# ==============================================================================
s = new_slide("Why Naive LLM Agents Fail in Autonomous ML Engineering", title_size=21.5)
pic(s, A / "svg_01_failure_modes.png", 0.35, 1.06, w=9.30)
card_box(
    s, 0.35, 3.72, 9.30, 1.24,
    "Core Design Thesis: Principled Evolutionary Search Inside Deterministic Contracts",
    [
        [("1. Principled Search Over Blind Retries:  ", {"bold": True, "color": BLUE}),
         ("Orthogonal multi-family seeding, UCT-to-elite scheduling, 7 lifecycle operators, and contrastive (+1/-1) RRF memory prevent branch collapse.", {})],
        [("2. Deterministic Contracts Over Self-Reporting:  ", {"bold": True, "color": TEAL}),
         ("Every candidate is independently re-scored on val.csv, audited for leakage, and accepted only by a stratified two-half OOS gate.", {})],
    ],
    fill=PALEB, line=BLUE, title_color=NAVY, title_size=13.3, body_size=11.7,
)
footer(s, "Vector diagram: assets/svg/svg_01_failure_modes.svg · Grounded in empirical traces from 6h & 1h multi-GPU runs.")
notes(s, """
[Slide 2 | 0:25–1:00 | 35s]
In multi-hour ML runs, naive LLM agents fail in four predictable ways. First, data and validation leakage—such as pre-split scaling or tuning blend weights directly on validation labels—inflates scores that collapse on test sets. Second, greedy single-branch exploration stalls in local optima while parallel workers repeat dead ends. Third, timeouts calibrated on quick baselines kill deeper five-fold GBDTs and neural nets. Fourth, hard-label voting discards confidence and overfits validation splits. Every SuperAgent subsystem directly solves one failure mode.
""")


# ==============================================================================
# SLIDE 3 · End-to-End Architecture (Adapter <-> Core 1.1) (1:00–1:35 | 35s)
# ==============================================================================
s = new_slide("System Architecture: Strict Adapter–Core 1.1 Separation", title_size=21.5)
pic(s, A / "svg_02_architecture.png", 0.38, 1.03, w=9.24)
card_box(
    s, 0.35, 4.14, 9.30, 0.84,
    "Three Non-Negotiable Architectural Boundaries",
    [
        [("Physical Label Firewall: ", {"bold": True, "color": RED}),
         ("private/test_labels.csv stays outside task_dir;   ", {}),
         ("Task-Agnostic Core: ", {"bold": True, "color": BLUE}),
         ("zero dataset-specific rules in superagent/;   ", {}),
         ("Offline Replay Gate: ", {"bold": True, "color": TEAL}),
         ("core11.py verifies byte-exact combine.py replay without executing Core.", {})],
    ],
    fill=LIGHT, line=BORDER, title_color=NAVY, title_size=12.8, body_size=11.5,
)
footer(s, "Vector diagram: assets/svg/svg_02_architecture.svg · Contract schema v1.1 verified across 541 offline unit & contract tests.")
notes(s, """
[Slide 3 | 1:00–1:35 | 35s]
To guarantee zero leakage by construction, we decouple the project into two layers joined strictly by our Core 1.1 filesystem contract. On the left, the Course Adapter isolates private test labels outside the agent workspace behind a physical firewall. In the center, Core 1.1 runs four parallel workers across GPUs: profiling modality, seeding orthogonal families, driving the evolutionary search tree with hybrid memory, and building a gated final bundle. On the right, core11.py replays combine.py byte-for-byte offline before grading.
""")


# ==============================================================================
# SLIDE 4 · Search Engine I: Multi-Branch Tree, UCT-Elite & Chain Control (1:35–2:15 | 40s)
# ==============================================================================
s = new_slide("Search Engine I: Multi-Branch Tree, UCT–Elite & Chain Control", title_size=20.8)
pic(s, A / "svg_03_search_tree.png", 0.35, 1.02, w=9.30)
card_box(
    s, 0.35, 3.74, 4.55, 1.24,
    "Dual-Phase Explore -> Exploit (selection.py)",
    [
        [("• explore_weight(p): ", {"bold": True, "color": BLUE}), ("1.0 (<0.5) -> linear -> 0.2 (>=0.7)", {})],
        [("• UCT tree_walk: ", {"bold": True, "color": NAVY}), ("c(t) decays 1.414 -> 0.50", {})],
        [("• elite_pick: ", {"bold": True, "color": TEAL}), ("w ~ 1/rank + chain_soft_switch", {})],
    ],
    fill=PALEB, line=BLUE, title_color=NAVY, title_size=12.4, body_size=10.5, space_after=1.5,
)
card_box(
    s, 5.10, 3.74, 4.55, 1.24,
    "Smooth Reward & Concurrency Safety (chain.py)",
    [
        [("• Smooth reward: ", {"bold": True, "color": PURPLE}), ("min(1.5, 0.75 + g/max(|prev|,0.05))", {})],
        [("• Local chain state: ", {"bold": True, "color": DARK}), ("per-node no_gain_count (no race)", {})],
        [("• Anti-race plateau: ", {"bold": True, "color": GOLD}), ("_after_best_finished + exemption", {})],
    ],
    fill=PALEO, line=GOLD, title_color=GOLD, title_size=12.4, body_size=10.5, space_after=1.5,
)
footer(s, "Vector diagram: assets/svg/svg_03_search_tree.svg · Implemented in core/superagent/selection.py, chain.py & scheduling.py.")
notes(s, """
[Slide 4 | 1:35–2:15 | 40s]
Let us look closely at our search engine. At timestamp zero, orthogonal draft hypotheses seed distinct model families concurrently—for example, TabICL, LightGBM, and CatBoost on tabular tasks, or ConvNeXt and DINOv2 on vision. Exploration uses UCT with a decaying constant from one-point-four-one-four down to zero-point-five. Past fifty percent progress, explore_weight smoothly shifts toward elite_pick, which samples a branch-diversified top pool inversely by rank using thread-local state. Along chains, smooth step rewards up to one-point-five favor true breakthroughs, while per-node no-gain counters, forced root returns, and finish-time plateau filters eliminate multi-worker race conditions.
""")


# ==============================================================================
# SLIDE 5 · Search Engine II: 7 Evolutionary Operators & Adaptive Admission (2:15–2:55 | 40s)
# ==============================================================================
s = new_slide("Search Engine II: 7 Operators & Value-Density Admission", title_size=21.0)
pic(s, A / "svg_04_operators_adaptive.png", 0.35, 1.02, w=9.30)
card_box(
    s, 0.35, 3.74, 4.55, 1.24,
    "7 Lifecycle Operators & AST Patching",
    [
        [("• Local & repair: ", {"bold": True, "color": TEAL}), ("AST import-aware SEARCH/REPLACE", {})],
        [("• Escalate & resume: ", {"bold": True, "color": PURPLE}), ("10-step retrospect + zero-LLM promote", {})],
        [("• Cross-branch: ", {"bold": True, "color": BLUE}), ("borrow (p>=0.5) & Root synthesis", {})],
    ],
    fill=PALET, line=TEAL, title_color=TEAL, title_size=12.4, body_size=10.5, space_after=1.5,
)
card_box(
    s, 5.10, 3.74, 4.55, 1.24,
    "Complementarity Value-Density Admission",
    [
        [("• Value density: ", {"bold": True, "color": BLUE}), ("priority = merit * p_ok / max(need, 1)", {})],
        [("• Error diversity: ", {"bold": True, "color": TEAL}), ("bonus rewards fixing global-best errors", {})],
        [("• Slot timeout: ", {"bold": True, "color": NAVY}), ("max(2.5*q0.8(ok), 0.75*slot_share)", {})],
    ],
    fill=PALEB, line=BLUE, title_color=NAVY, title_size=12.4, body_size=10.5, space_after=1.5,
)
footer(s, "Vector diagram: assets/svg/svg_04_operators_adaptive.svg · Implemented in operators.py, codegen.py & control/adaptive.py.")
notes(s, """
[Slide 5 | 2:15–2:55 | 40s]
Across these branches, seven operators coordinate the lifecycle. Beyond initial drafts, refine applies AST-scoped patches that also allow top-level imports, repair fixes failures from tracebacks, and promote resumes saved checkpoints with zero LLM calls via Successive Halving. When branch or global plateaus occur past fifty percent progress, retrospect reviews ten-step trajectories while borrow and synthesis recombine complementary branches. Crucially, AdaptiveController ranks multi-slot proposals by value density, combining rank strength, child novelty, validation error complementarity against the global best, and a decoupled slot-scaled timeout floor.
""")


# ==============================================================================
# SLIDE 6 · Hybrid RRF Experience Store & Method Priors (2:55–3:30 | 35s)
# ==============================================================================
s = new_slide("Hybrid Knowledge Base: Qwen3 + CamelCase BM25 + RRF", title_size=20.2)
pic(s, A / "svg_05_hybrid_memory.png", 0.35, 1.04, w=9.30)
card_box(
    s, 0.35, 3.74, 9.30, 1.22,
    "Why Contrastive Hybrid Reciprocal Rank Fusion Beats Single-Channel Memory",
    [
        [("• Complementary Recall:  ", {"bold": True, "color": PURPLE}),
         ("Local CPU Qwen3-Embedding-0.6B (1024-d, zero GPU contention) captures ML concepts; CamelCase+CJK Okapi BM25 matches exact symbols.", {})],
        [("• Contrastive Anti-Looping:  ", {"bold": True, "color": TEAL}),
         ("RRF (k=60) injects Top-2 positive (+1) gains & Top-2 negative (-1) counterexamples (prioritizing failed children) into planning.", {})],
    ],
    fill=PALET, line=TEAL, title_color=TEAL, title_size=12.6, body_size=11.0, space_after=1.8,
)
footer(s, "Vector diagram: assets/svg/svg_05_hybrid_memory.svg · Implemented in core/superagent/experience.py & catalog.py.")
notes(s, """
[Slide 6 | 2:55–3:30 | 35s]
To stop parallel workers from repeating failed experiments, we built a hybrid ExperienceStore. Every evaluated node logs its structured plan, validation effect, and traceback. At planning time, Reciprocal Rank Fusion merges dense semantic embeddings from local CPU Qwen3-0.6B with a CamelCase- and CJK-aware Okapi BM25 index over exact API symbols like StratifiedKFold or CUDA errors. Feeding the top two positive gains alongside top two negative counterexamples—prioritizing direct failed children—guides autonomous component targeting cleanly.
""")


# ==============================================================================
# SLIDE 7 · Validated Ensembling & Probability Sidecars (3:30–4:05 | 35s)
# ==============================================================================
s = new_slide("Trustworthy Finalization: Stratified Two-Half Gate & Prob Sidecars", title_size=20.4)
pic(s, A / "svg_06_validated_ensemble.png", 0.35, 1.04, w=9.30)
card_box(
    s, 0.35, 3.56, 9.30, 1.38,
    "Why Calibrated Probability Sidecars and Stratified Out-of-Sample Gating Are Essential",
    [
        [("• Closing Hard-Label Confidence Loss:  ", {"bold": True, "color": BLUE}),
         ("Contract predictions.csv stores discrete labels, discarding confidence. Optional val/test_probabilities.csv sidecars and +2 odd-pair greedy search restore soft fusion.", {})],
        [("• stratified_halves Zero-Harm Gate:  ", {"bold": True, "color": GOLD}),
         ("Weights fit on stratified Half A must beat best single on Half B AND vice versa (or win log_loss > 0.001 on ties)—else keeps best single.", {})],
    ],
    fill=PALEB, line=BLUE, title_color=NAVY, title_size=12.8, body_size=11.2,
)
footer(s, "Vector diagram: assets/svg/svg_06_validated_ensemble.svg · Implemented in core/superagent/final.py, ensemble.py & metrics.py.")
notes(s, """
[Slide 7 | 3:30–4:05 | 35s]
During finalization, naive agents either blind-average top scripts or tune weights directly on validation labels. SuperAgent applies a deterministic stratified_halves out-of-sample gate across diverse top candidates: weights fit on stratified Half A must beat the single best model on disjoint Half B, and vice versa. Because discrete labels drop confidence, optional probability sidecars, odd-pair greedy search, and a cross-entropy log-loss tie-breaker enable soft probability blending with byte-exact offline replay.
""")


# ==============================================================================
# SLIDE 8 · Evaluation I: Multi-Modality Vision & Search Health (4:05–4:35 | 30s)
# ==============================================================================
s = new_slide("Evaluation I: Multi-Modality Vision & Multi-Branch Search Health", title_size=20.4)
pic(s, A / "plot_02_multimodal_search.png", 0.55, 1.02, h=2.92)
card_box(
    s, 0.35, 3.98, 4.55, 0.98,
    "Intel Image 6-Class Vision (14k Train / 3k Test)",
    [
        [("• Validated top6 ensemble: ", {"bold": True, "color": TEAL}),
         ("95.33% test acc (95.45% macro-F1), beating 95.13% single across 5 branches.", {"bold": True})],
    ],
    fill=PALET, line=TEAL, title_color=TEAL, title_size=12.6, body_size=11.3,
)
card_box(
    s, 5.10, 3.98, 4.55, 0.98,
    "Cross-Modality Search Execution Health",
    [
        [("• Sustained valid yield: ", {"bold": True, "color": NAVY}),
         ("84.7% valid on Vision (72/85); rose 65.0% -> 86.0% on Tabular w/ 11 promotes.", {})],
    ],
    fill=LIGHT, line=NAVY, title_color=NAVY, title_size=12.6, body_size=11.3,
)
footer(s, "Plot: assets/plot_02_multimodal_search.png · Runs on 4× RTX 4090 GPUs; task-agnostic Core 1.1 with sealed private/ grading.")
notes(s, """
[Slide 8 | 4:05–4:35 | 30s]
We evaluated SuperAgent without changing a single core line across vision and tabular benchmarks on RTX 4090 GPUs. On the six-class Intel Image dataset with three thousand sealed test photos, SuperAgent explored eighty-five nodes across five diverse branches including ConvNeXt, Swin, and EVA-02. Our stratified two-half gate accepted a six-model ensemble scoring ninety-five point three-three percent private test accuracy, beating the ninety-five point one-three single model with eighty-four point seven percent valid node execution.
""")


# ==============================================================================
# SLIDE 9 · Evaluation II: Breaking the 25s Timeout Spiral (4:35–5:05 | 30s)
# ==============================================================================
s = new_slide("Evaluation II: Eliminating the Timeout Spiral & Metric Drift", title_size=21.0)
pic(s, A / "plot_03_timeout_fix.png", 0.65, 1.06, w=8.70)
card_box(
    s, 0.35, 3.86, 9.30, 1.12,
    "Closed-Loop Search Diagnosis -> Verified Fix on 80k-Row Airline Benchmark",
    [
        [("• Execution Stability: ", {"bold": True, "color": TEAL}),
         ("Slot-scaled kill ceiling cut timeout kills from 31/120 (25.8%) to 1/57 (1.8%); valid node rate jumped 65.0% -> 86.0%.", {})],
        [("• Contract Alignment: ", {"bold": True, "color": BLUE}),
         ("Task-hint disambiguation + strict val.csv rule closed the 0.37% OOF gap to 0.00% (92.94% == 92.94%).", {})],
    ],
    fill=PALEB, line=BLUE, title_color=NAVY, title_size=12.6, body_size=11.1, space_after=1.5,
)
footer(s, "Plot: assets/plot_03_timeout_fix.png · Playground S6E10 Airline Satisfaction (70k train / 10k val / 20k sealed test).")
notes(s, """
[Slide 9 | 4:35–5:05 | 30s]
Telemetry on the eighty-thousand-row Airline tabular benchmark exposed two structural search bugs. First, fast fifteen-second drafts compressed quantile timeouts to twenty-five seconds, killing thirty-one deeper GBDT and neural nodes. Decoupling expected duration from our slot-scaled kill ceiling reduced timeout kills from thirty-one to one, lifting valid yield to eighty-six percent. Second, metric disambiguation and shared validation alignment eliminated a zero-point-three-seven percent OOF mismatch, passing every contract audit cleanly.
""")


# ==============================================================================
# SLIDE 10 · Controlled 1h Head-to-Head vs MLEvolve (5:05–5:40 | 35s)
# ==============================================================================
s = new_slide("Head-to-Head: SuperAgent vs. MLEvolve (4× RTX 4090, 1h)", title_size=21.0)
h2h_rows = [
    ["Metric (Identical 1h / 4-GPU Setup)", "SuperAgent (v1)", "MLEvolve (1h)"],
    ["Evaluated Nodes (Valid)", "57 (49 ok)", "30 (27 ok)"],
    ["Zero-LLM Rung Promotes", "11 (ckpt resume)", "0 (LLM regen)"],
    ["Search Throughput (Time)", "1.9× (53.6 min)", "1.0× (60.0 min)"],
    ["Primary Delivered Test Acc", "92.175% (single)", "92.170% (single)"],
    ["Best Single Discovered (Test)", "92.245% (rank_06)", "92.235% (top3)"],
    ["Ensemble Governance", "Stratified OOS Gate", "Unvalidated Prob Avg"],
]
table(s, h2h_rows, 0.35, 1.06, 5.05, [2.25, 1.42, 1.38], size=10.6, row_h=0.36, hl_rows=(1, 2, 3, 5))
pic(s, A / "plot_04_head2head.png", 5.52, 1.06, w=4.13)
card_box(
    s, 0.35, 3.80, 9.30, 1.18,
    "Key Head-to-Head Insight: Higher Search Throughput & Stronger Models + Closing the Fusion Gap",
    [
        [("• Search Advantage: ", {"bold": True, "color": TEAL}),
         ("1.9× search throughput via checkpoint promote & AdaptiveController; won primary (92.175%) and best single (92.245%) test acc.", {})],
        [("• Architectural Lesson: ", {"bold": True, "color": GOLD}),
         ("MLEvolve's prob blend (92.205%) exposed discrete vote loss—motivating our calibrated sidecars & stratified OOS gate!", {})],
    ],
    fill=LIGHT, line=NAVY, title_color=NAVY, title_size=12.2, body_size=10.6, space_after=1.4,
)
footer(s, "Identical hardware & split: 80,000 dev rows / 20,000 sealed test rows, 4× RTX 4090, 16 CPUs, 4 workers, deepseek-flash.")
notes(s, """
[Slide 10 | 5:05–5:40 | 35s]
Next, we ran a controlled one-hour benchmark against MLEvolve on the exact same eighty-thousand-row Airline split, four RTX 4090 GPUs, and DeepSeek backend. Thanks to AdaptiveController value-density scheduling and eleven zero-LLM checkpoint promotions, SuperAgent evaluated fifty-seven nodes versus MLEvolve's thirty—nearly double the search throughput. On the twenty-thousand-row test set, SuperAgent achieved stronger primary accuracy at ninety-two point one-seven-five percent and best single model at ninety-two point two-four-five percent, while MLEvolve's probability blend motivated our sidecar upgrade.
""")


# ==============================================================================
# SLIDE 11 · Task-Agnostic Generalizations & Engineering Hardening (5:40–6:10 | 30s)
# ==============================================================================
s = new_slide("Task-Agnostic Generalization & Concurrency Hardening", title_size=21.5)
pic(s, A / "svg_07_generalization.png", 0.35, 1.06, w=9.30)
card_box(
    s, 0.35, 3.54, 9.30, 1.42,
    "Strict Generalization Principle & Integrated Engineering Hardening",
    [
        [("• Zero Dataset Hardcoding:  ", {"bold": True, "color": BLUE}),
         ("Upgrades operate strictly on contract schemas, validation splits, AST structure, and search history across tabular, vision, text, and multimodal tasks.", {})],
        [("• Hardened Search & Memory:  ", {"bold": True, "color": PURPLE}),
         ("Local no_gain_count, _after_best_finished plateau filter, slot rank_proposals, AST import patching, CamelCase+CJK BM25, stratified_halves.", {})],
        [("• Full Regression Verification:  ", {"bold": True, "color": TEAL}),
         ("410 Core unit tests + 131 Course Adapter contract tests = 541 passed, 0 failed (offline suite).", {"bold": True})],
    ],
    fill=PALET, line=TEAL, title_color=TEAL, title_size=12.6, body_size=10.7, space_after=1.5,
)
footer(s, "Vector diagram: assets/svg/svg_07_generalization.svg · Verified offline via pytest (541 passed, 7 skipped live-LLM checks).")
notes(s, """
[Slide 11 | 5:40–6:10 | 30s]
Inspecting our strongest scripts led directly to four generalizable upgrades across all modalities: concurrency-safe per-node chain state and plateau filtering, AST import-aware scoped patching with autonomous target selection and CamelCase BM25 memory, stratified two-half validation with probability sidecars and anti-snooping audits, plus orthogonal seeding and slot-scaled timeouts. Together, our Core and Adapter repositories pass all five hundred and forty-one automated offline unit and contract tests with zero failures.
""")


# ==============================================================================
# SLIDE 12 · Conclusion: Three Core Design Principles (6:10–6:30 | 20s)
# ==============================================================================
s = new_slide("Conclusion: Three Design Principles for Autonomous ML Search", title_size=20.8)
card_box(
    s, 0.35, 1.10, 2.96, 2.50,
    "1 · Contract-First Zero-Leakage Isolation",
    [
        "• Physical private/ label firewall outside task_dir",
        "• Independent val.csv scoring & AST/risk leak audit",
        "• Pure offline core11.py byte-exact replay",
    ],
    fill=PALEB, line=BLUE, title_color=BLUE, title_size=13.0, body_size=11.6, space_after=3.5,
)
card_box(
    s, 3.52, 1.10, 2.96, 2.50,
    "2 · Principled Evolutionary Tree Search",
    [
        "• Orthogonal seeding + UCT <-> Elite 7-op search",
        "• Complementarity value-density & slot timeouts",
        "• Contrastive Qwen3 + CamelCase BM25 RRF memory",
    ],
    fill=PALET, line=TEAL, title_color=TEAL, title_size=13.0, body_size=11.6, space_after=3.5,
)
card_box(
    s, 6.69, 1.10, 2.96, 2.50,
    "3 · Stratified Out-of-Sample Ensembling",
    [
        "• Task-hint + 1e-4 exact-hit metric identification",
        "• Calibrated probability sidecars + odd-pair greedy",
        "• stratified_halves OOS gate + log_loss tie-break",
    ],
    fill=PALEO, line=GOLD, title_color=GOLD, title_size=13.0, body_size=11.6, space_after=3.5,
)
card_box(
    s, 0.35, 3.74, 9.30, 1.22,
    "Headline Empirical Summary & Future Direction",
    [
        [("• Verified Outcomes:  ", {"bold": True, "color": NAVY}),
         ("95.33% Intel Image test acc (top6 > 95.13% single) · 1.9× search throughput vs MLEvolve · 92.245% best Airline single · 541/541 offline tests.", {})],
        [("• Next Milestone:  ", {"bold": True, "color": TEAL}),
         ("Cross-branch second-stage OOF stacking operator & anonymized method distillation. Thank you — welcome your questions!", {"bold": True})],
    ],
    fill=LIGHT, line=NAVY, title_color=NAVY, title_size=12.6, body_size=11.0, space_after=1.8,
)
footer(s, "SuperAgent · AIAA 3102 Course Project · HKUST(GZ) · Q&A Backup Slides B1–B4 follow.")
notes(s, """
[Slide 12 | 6:10–6:30 | 20s]
In summary, SuperAgent unifies three core design principles: strict contract isolation against data leakage, adaptive multi-branch evolutionary tree search guided by complementarity and contrastive hybrid memory, and stratified out-of-sample ensembling—achieving ninety-five point three-three percent vision accuracy, nearly double MLEvolve's search throughput, and five hundred and forty-one passing tests. Thank you for listening, and we welcome your questions!
""")


# ==============================================================================
# SLIDE 13 · Backup Section Header (Un-timed, for Q&A Defense)
# ==============================================================================
sb_div = prs.slides.add_slide(SECTION_LAYOUT)
sb_div.shapes.title.text = "Backup Slides for Deep Q&A (B1–B4)"
r = sb_div.shapes.title.text_frame.paragraphs[0].runs[0]
r.font.color.rgb = NAVY
r.font.bold = True
r.font.name = FONT
for sh in TPL_CONTENT.shapes:
    if sh.is_placeholder and sh.placeholder_format.idx == 12:
        sb_div.shapes._spTree.append(copy.deepcopy(sh._element))
tb(sb_div, 0.50, 3.20, 9.0, 1.5, [
    "• B1: Four-Layer Anti-Leakage Architecture & Offline core11.py Byte-Exact Replay",
    "• B2: Search, Scheduling, RRF Memory & Stratified OOS Mathematical Formulations",
    "• B3: Engineering Hygiene Appendix — Role-Stratified Thinking & Optional Budget Guardrails",
    "• B4: Full Architectural Matrix (SuperAgent vs. MLEvolve & Integrated Hardening)",
], size=14.0, color=DARK, spacing=6)
notes(sb_div, "[Backup Divider | Q&A Only] Navigate directly to B1–B4 during roadshow Q&A defense.")


# ==============================================================================
# SLIDE 14 · B1: Anti-Leakage Defense & Offline core11.py Audit
# ==============================================================================
s = new_slide("B1 · Four-Layer Anti-Leakage Defense & Offline core11 Replay", title_size=20.5)
b1_rows = [
    ["Defense Layer", "Module / Boundary", "What Threat It Blocks", "Verification Mechanism"],
    ["L1: Physical Label Firewall", "adapter: prepare & task_dir", "Reading test answers or leaky columns", "private/ outside task_dir; SHA-256 & schema lock"],
    ["L2: Static AST & Risk Review", "prompts/ + review.py", "Pre-split scaling, val.csv weight tuning", "Regex risk gate triggers LLM review before GPU run"],
    ["L3: Runtime Metric & Jump Audit", "assessment.py & metrics.py", "Stdout spoofing, OOF vs val.csv drift", "Re-scores val.csv (1e-4 hit) + jump leak audit"],
    ["L4: Offline Contract Replay", "course/back/core11.py", "Un-reproducible delivery or arbitrary code", "Zero Core imports; checks symlinks & byte-exact CSV"],
]
table(s, b1_rows, 0.35, 1.08, 9.30, [1.95, 1.95, 2.45, 2.95], size=11.0, row_h=0.54, hl_rows=(1, 4))
card_box(
    s, 0.35, 4.02, 9.30, 0.92,
    "Key Q&A Defense Point: Can combine.py cheat or import arbitrary dependencies during grading?",
    [
        "No: final.py emits a pure-Python CSV combiner (vote or prob_argmax), and course/back/core11.py independently recomputes the ensemble in memory and asserts byte-equality against final/predictions.csv.",
    ],
    fill=PALET, line=TEAL, title_color=TEAL, title_size=12.6, body_size=11.5,
)
footer(s, "Backup B1 · Jump here for Q&A on data leakage, validation overfitting, or Adapter–Core trust boundaries.")
notes(s, "[Backup B1 | Q&A Defense] Explains our four-layer defense-in-depth against test leakage, preprocessing leakage, validation snooping, and delivery spoofing.")


# ==============================================================================
# SLIDE 15 · B2: Core Search, Memory & Ensembling Mathematics
# ==============================================================================
s = new_slide("B2 · Core Mathematical Formulations (Search, Admission & Gate)", title_size=20.2)
card_box(
    s, 0.35, 1.06, 4.52, 2.04,
    "1. Search Tree & Smooth Reward (selection.py, chain.py)",
    [
        [("• UCB: ", {"bold": True, "color": BLUE}), ("R_c/N_c + c(t)*sqrt(ln N_p / N_c); c: 1.414 -> 0.5", {})],
        [("• Explore prob: ", {"bold": True, "color": NAVY}), ("1.0 (<0.5) -> linear decay -> 0.2 (>=0.7)", {})],
        [("• Step reward: ", {"bold": True, "color": PURPLE}), ("min(1.5, 0.75 + g / max(|s_prev|, 0.05))", {"bold": True})],
        "• Fail = -1.0; first ok = +0.5; g < 1e-4 = 0.0; local no_gain <= 3",
    ],
    fill=PALEB, line=BLUE, title_color=NAVY, title_size=12.2, body_size=10.8, space_after=2,
)
card_box(
    s, 5.13, 1.06, 4.52, 2.04,
    "2. Value-Density & Slot Timeout (adaptive.py)",
    [
        [("• Priority: ", {"bold": True, "color": TEAL}), ("merit * p_ok / max(need_secs, 1)", {"bold": True})],
        "• merit = 0.2 + 1/(1+0.12r) + 0.4/(1+ch) + min(0.5, comp)",
        [("• Error comp: ", {"bold": True, "color": GOLD}), ("mean((best != y_val) & (node == y_val))", {})],
        "• Kill = clip(max(2.5*q0.8(ok), 0.75*slot_share), min, 0.5*usable)",
    ],
    fill=PALET, line=TEAL, title_color=TEAL, title_size=12.2, body_size=10.8, space_after=2,
)
card_box(
    s, 0.35, 3.22, 9.30, 1.72,
    "3. Hybrid RRF Memory & stratified_halves Out-of-Sample Gate (experience.py, final.py)",
    [
        [("• Hybrid RRF (k=60):  ", {"bold": True, "color": PURPLE}),
         ("RRF(i) = 0.5/(60 + r_kw) + 0.5/(60 + r_vec), merging Qwen3-Embedding-0.6B (cos>=0.35) + CamelCase/CJK BM25.", {})],
        [("• Two-Half Gate:  ", {"bold": True, "color": GOLD}),
         ("stratified_halves splits val into Half A & Half B -> fit on A, score B & fit on B, score A -> accept iff BOTH improve by >= 1e-4.", {})],
        [("• Sidecar Calibration Tie-Break:  ", {"bold": True, "color": TEAL}),
         ("On metric ties (<1e-9), accept iff val_probabilities.csv log_loss drops by > 0.001 on BOTH halves.", {})],
    ],
    fill=PALEO, line=GOLD, title_color=GOLD, title_size=12.6, body_size=11.1, space_after=2,
)
footer(s, "Backup B2 · Exact mathematical equations and thresholds verified directly in core/superagent/ sources.")
notes(s, "[Backup B2 | Q&A Defense] Exact equations for UCT decay, smooth step reward, complementarity merit, slot_share timeout, weighted RRF, and stratified_halves OOS gating.")


# ==============================================================================
# SLIDE 16 · B3: Engineering Hygiene Appendix (Thinking Policy & Ledger)
# ==============================================================================
s = new_slide("B3 · Engineering Appendix: Role-Stratified CoT & Cost Hygiene", title_size=20.0)
b3_rows = [
    ["Operational Hygiene Metric", "Airline 6h (v0 Baseline)", "Intel Image 6h (v0)", "Airline 1h (v1 Stratified)"],
    ["Search Runtime / Stop Trigger", "89.4 min (warn cap)", "132.6 min (warn cap)", "53.6 min (reserve_stop)"],
    ["Total Reasoning (CoT) Tokens", "2,358,966 (52.1% spend)", "2,411,734 (53.1% spend)", "735,416 (-68.8% reduction)"],
    ["patch_coder Reasoning Policy", "Unthrottled (~14.9k/call)", "Unthrottled (~1.8M total)", "0 CoT tokens (CoT OFF)"],
    ["reviewer Policy", "Broad regex (94.7% CoT)", "Broad regex (96.9% CoT)", "Risk-gated (split/val-tune only)"],
    ["planner.structured / draft", "Deep Thinking ON", "Deep Thinking ON", "Deep Thinking ON (preserved!)"],
    ["Pre-Send Ledger & Guardrail", "SQLite WAL + 0.9 warn", "SQLite WAL + 0.9 warn", "Optional soft cap; never interrupts GPU"],
]
table(s, b3_rows, 0.35, 1.08, 9.30, [2.55, 2.25, 2.25, 2.25], size=10.8, row_h=0.43, hl_rows=(2, 3, 5))
card_box(
    s, 0.35, 4.22, 9.30, 0.74,
    "Why Keep This Secondary: Supports Search Efficiency Without Sacrificing Reasoning Depth",
    [
        "Full CoT stays active on planner/drafts; stripping mechanical patch_coder CoT lowered latency (helping reach 1.9× throughput), while optional SQLite pre-send reserves cap spend without aborting active GPUs.",
    ],
    fill=LIGHT, line=NAVY, title_color=NAVY, title_size=11.6, body_size=10.3, space_after=1.0,
)
footer(s, "Backup B3 · Supporting operational telemetry (operations.sqlite + delivery/api_usage.json); secondary to search design.")
notes(s, "[Backup B3 | Q&A Defense] Engineering hygiene summary explaining how role-stratified thinking preserved full planning quality while cutting mechanical CoT latency, and how optional pre-send accounting works safely under parallel workers.")


# ==============================================================================
# SLIDE 17 · B4: Architectural Comparison Matrix & Integrated Hardening
# ==============================================================================
s = new_slide("B4 · SuperAgent vs. MLEvolve Architecture & Integrated Hardening", title_size=20.0)
b4_rows = [
    ["Core Architecture Dimension", "MLEvolve", "SuperAgent (Ours)", "Search & Design Advantage"],
    ["Label & Workspace Boundary", "Shared workspace conventions", "Adapter <-> Core 1.1 + core11.py", "Physical label isolation & offline replay"],
    ["Root Seeding & Diversity", "Serial ~13m initial drafts", "Concurrent orthogonal hypotheses", "Parallel 4-GPU launch with family spread"],
    ["Tree & Plateau Dynamics", "Sequential tree expansion", "UCT <-> Elite + 7 ops + local state", "Concurrency-safe no_gain & _after_best filter"],
    ["Offer Admission Priority", "Heuristic queue", "Complementarity value density + slots", "Rewards validation errors fixed over best"],
    ["Experience & Code Editing", "Pure dense RAG + full rewrite", "Qwen3 + CamelCase BM25 RRF + AST patch", "Contrastive (+1/-1) recall + scoped imports"],
    ["Final Ensemble Verification", "Unvalidated top-K prob avg", "stratified_halves OOS + prob sidecars", "Soft confidence blend + zero-harm fallback"],
]
table(s, b4_rows, 0.35, 1.08, 9.30, [2.05, 2.10, 2.60, 2.55], size=10.8, row_h=0.41, hl_rows=(2, 3, 4, 6))
card_box(
    s, 0.35, 4.16, 9.30, 0.80,
    "Selective Hardening Integration Summary (541 Passing Offline Tests)",
    [
        "Integrated generalizable concurrency & search safeguards (local no_gain_count, _after_best_finished, smooth step_reward, slot rank_proposals, AST import patching, CamelCase+CJK BM25, stratified_halves); rejected unvalidated auto-ensembles.",
    ],
    fill=PALEB, line=BLUE, title_color=NAVY, title_size=11.6, body_size=10.2, space_after=1.0,
)
footer(s, "Backup B4 · Jump here for Q&A comparing SuperAgent with AIDE / MLEvolve or discussing the 1121 hardening integration.")
notes(s, "[Backup B4 | Q&A Defense] Detailed architectural comparison against MLEvolve on search tree mechanics, admission, memory, patching, and validated ensembling.")


# ==============================================================================
# Finalize: Remove Template Sample Slides [1..9] & Renumber Slide Fields 1..N
# ==============================================================================
sldIdLst = prs.slides._sldIdLst
for sldId in list(sldIdLst)[1:10]:
    prs.part.drop_rel(sldId.rId)
    sldIdLst.remove(sldId)

for k, sl in enumerate(prs.slides, 1):
    for fld in sl._element.iter(qn("a:fld")):
        if fld.get("type") == "slidenum":
            fld.find(qn("a:t")).text = str(k)

prs.save(OUT)
print(f"Saved {OUT} ({len(prs.slides)} slides total: 12 core + 1 divider + 4 backup)")
