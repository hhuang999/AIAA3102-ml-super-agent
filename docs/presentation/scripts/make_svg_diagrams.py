"""Generate 7 publication-grade W3C vector SVG diagrams for the SuperAgent presentation,
and render crisp Retina PNGs via macOS WebKit (qlmanage).
"""
import subprocess
import tempfile
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "assets"
SVG_DIR = A / "svg"
SVG_DIR.mkdir(parents=True, exist_ok=True)

COMMON_DEFS = """
  <defs>
    <filter id="sh" x="-4%" y="-6%" width="108%" height="118%">
      <feDropShadow dx="0" dy="3" stdDeviation="5" flood-color="#0F172A" flood-opacity="0.08"/>
    </filter>
    <linearGradient id="gNavyHdr" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#112B5C"/>
      <stop offset="100%" stop-color="#1E3A8A"/>
    </linearGradient>
    <linearGradient id="gBlueHdr" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#1D4ED8"/>
      <stop offset="100%" stop-color="#2563EB"/>
    </linearGradient>
    <linearGradient id="gTealHdr" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#0F766E"/>
      <stop offset="100%" stop-color="#0D9488"/>
    </linearGradient>
    <linearGradient id="gGoldHdr" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#92400E"/>
      <stop offset="100%" stop-color="#B45309"/>
    </linearGradient>
    <linearGradient id="gPurpleHdr" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#5B21B6"/>
      <stop offset="100%" stop-color="#7C3AED"/>
    </linearGradient>
    <linearGradient id="gRedHdr" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#991B1B"/>
      <stop offset="100%" stop-color="#DC2626"/>
    </linearGradient>
    <marker id="arrSlate" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
      <path d="M 0 1.2 L 9.5 5 L 0 8.8 z" fill="#475569"/>
    </marker>
    <marker id="arrBlue" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
      <path d="M 0 1.2 L 9.5 5 L 0 8.8 z" fill="#1D4ED8"/>
    </marker>
    <marker id="arrTeal" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
      <path d="M 0 1.2 L 9.5 5 L 0 8.8 z" fill="#0F766E"/>
    </marker>
    <marker id="arrGold" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
      <path d="M 0 1.2 L 9.5 5 L 0 8.8 z" fill="#B45309"/>
    </marker>
    <marker id="arrPurple" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
      <path d="M 0 1.2 L 9.5 5 L 0 8.8 z" fill="#6D28D9"/>
    </marker>
    <marker id="arrRed" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
      <path d="M 0 1.2 L 9.5 5 L 0 8.8 z" fill="#DC2626"/>
    </marker>
    <style>
      text { font-family: Arial, Helvetica, sans-serif; }
      .mono { font-family: Menlo, Consolas, monospace; }
    </style>
  </defs>
"""


def render_svg(name: str, w: int, h: int, body: str, px_size: int = 2600) -> Path:
    import re
    import xml.etree.ElementTree as ET
    body = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;)", "&amp;", body)
    clean_svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n{COMMON_DEFS}\n  <rect width="{w}" height="{h}" fill="#FFFFFF"/>\n{body}\n</svg>\n'
    ET.fromstring(clean_svg)
    svg_path = SVG_DIR / f"{name}.svg"
    svg_path.write_text(clean_svg, encoding="utf-8")

    wk_w = 760.0
    wk_h = wk_w * h / w
    wk_svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{wk_w:.3f}" height="{wk_h:.3f}" viewBox="0 0 {w} {h}">\n{COMMON_DEFS}\n  <rect width="{w}" height="{h}" fill="#FFFFFF"/>\n{body}\n</svg>\n'
    with tempfile.TemporaryDirectory() as td:
        tmp_svg = Path(td) / f"{name}.svg"
        tmp_svg.write_text(wk_svg, encoding="utf-8")
        subprocess.run(
            ["qlmanage", "-t", "-s", str(px_size), "-o", td, str(tmp_svg)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        im = Image.open(Path(td) / f"{name}.svg.png").convert("RGB")
        crop_h = round(im.width * h / w)
        im = im.crop((0, 0, im.width, crop_h))
        out_png = A / f"{name}.png"
        im.save(out_png, optimize=True)
    print(f"Rendered {svg_path.name} -> {out_png.name} ({im.width}x{im.height})")
    return out_png


def card(x, y, w, h, hdr_grad, hdr_text, bg, stroke, lines, hdr_h=42, hdr_fs=19.5):
    out = [
        f'<g filter="url(#sh)">',
        f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{bg}" stroke="{stroke}" stroke-width="2"/>',
        f'  <path d="M {x} {y+14} Q {x} {y} {x+14} {y} L {x+w-14} {y} Q {x+w} {y} {x+w} {y+14} L {x+w} {y+hdr_h} L {x} {y+hdr_h} Z" fill="url(#{hdr_grad})"/>',
        f'  <text x="{x + w/2}" y="{y + hdr_h/2 + 7}" text-anchor="middle" fill="#FFFFFF" font-size="{hdr_fs}" font-weight="bold">{hdr_text}</text>',
    ]
    cy = y + hdr_h + 28
    for txt, sz, col, bold, dy in lines:
        fw = "bold" if bold else "normal"
        out.append(f'  <text x="{x + 18}" y="{cy}" fill="{col}" font-size="{sz}" font-weight="{fw}">{txt}</text>')
        cy += dy
    out.append("</g>")
    return "\n".join(out)


# ==============================================================================
# 1. Slide 2: 4 Failure Modes of Naive LLM Agents vs SuperAgent Countermeasures
# ==============================================================================
def make_svg_01_failure_modes():
    W, H = 1840, 500
    cols = [
        ("1. Data & Validation Leakage",
         ["Problem:", "• Pre-split scaling / target encoding", "• Tuning blend weights on val.csv", "• Unsealed test answer columns",
          "SuperAgent Remedy:", "✓ Physical private/ label firewall", "✓ Inner-CV only weight/threshold rule", "✓ AST & jump-triggered leak audit"]),
        ("2. Greedy Search & Branch Collapse",
         ["Problem:", "• Single-family drafts lack diversity", "• Naive hill-climbing stalls locally", "• Parallel workers repeat failed ideas",
          "SuperAgent Remedy:", "✓ Orthogonal family seeding (t = 0)", "✓ UCT &lt;-&gt; Elite + 7-op search tree", "✓ Contrastive (+1/-1) RRF memory"]),
        ("3. Script Timeout Death Spiral",
         ["Problem:", "• Fast 15s baseline shrinks q0.8 limit", "• 20–32s cap kills 5-fold GBDT/NNs", "• 31/120 timeouts trigger repair loops",
          "SuperAgent Remedy:", "✓ Decouple priority vs kill ceiling", "✓ Crash-immune ok_script estimator", "✓ Slot-share floor (31 -> 1 timeout)"]),
        ("4. Fragile Metric & Blind Fusion",
         ["Problem:", "• Binary metric ties (acc vs prec_macro)", "• Hard 0/1 vote loses confidence", "• Unvalidated blending overfits val",
          "SuperAgent Remedy:", "✓ Task-hint + 1e-4 exact-hit disambig.", "✓ Stratified two-half OOS gate", "✓ Prob-sidecar + log_loss tie-break"]),
    ]
    parts = []
    cw, gap, x0 = 430, 26, 20
    for idx, (title, items) in enumerate(cols):
        x = x0 + idx * (cw + gap)
        parts.append(f'<g filter="url(#sh)">')
        parts.append(f'  <rect x="{x}" y="18" width="{cw}" height="462" rx="16" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="2"/>')
        parts.append(f'  <path d="M {x} 34 Q {x} 18 {x+16} 18 L {x+cw-16} 18 Q {x+cw} 18 {x+cw} 34 L {x+cw} 68 L {x} 68 Z" fill="url(#gNavyHdr)"/>')
        parts.append(f'  <text x="{x+cw/2}" y="50" text-anchor="middle" fill="#FFFFFF" font-size="19.5" font-weight="bold">{title}</text>')
        parts.append(f'  <rect x="{x+16}" y="84" width="{cw-32}" height="168" rx="10" fill="#FEF2F2" stroke="#FCA5A5" stroke-width="1.5"/>')
        y = 114
        for ln in items[:4]:
            bold = "bold" if ln.endswith(":") else "normal"
            col = "#991B1B" if ln.endswith(":") else "#334155"
            parts.append(f'  <text x="{x+28}" y="{y}" fill="{col}" font-size="18" font-weight="{bold}">{ln}</text>')
            y += 34
        parts.append(f'  <line x1="{x+cw/2}" y1="256" x2="{x+cw/2}" y2="284" stroke="#0F766E" stroke-width="3" marker-end="url(#arrTeal)"/>')
        parts.append(f'  <rect x="{x+16}" y="292" width="{cw-32}" height="172" rx="10" fill="#F0FDFA" stroke="#5EEAD4" stroke-width="1.8"/>')
        y = 322
        for ln in items[4:]:
            bold = "bold" if ln.endswith(":") else "normal"
            col = "#0F766E" if ln.endswith(":") else "#1E293B"
            parts.append(f'  <text x="{x+28}" y="{y}" fill="{col}" font-size="18" font-weight="{bold}">{ln}</text>')
            y += 34
        parts.append("</g>")
    render_svg("svg_01_failure_modes", W, H, "\n".join(parts))


# ==============================================================================
# 2. Slide 3: System Architecture (Adapter <-> Core 1.1 Closed Loop)
# ==============================================================================
def make_svg_02_architecture():
    W, H = 1840, 600
    b = []
    b.append('<rect x="18" y="18" width="350" height="560" rx="16" fill="#F8FAFC" stroke="#94A3B8" stroke-width="2" stroke-dasharray="7,5"/>')
    b.append('<text x="193" y="48" text-anchor="middle" fill="#112B5C" font-size="21" font-weight="bold">1. Input Adapter Layer</text>')
    b.append(card(36, 68, 314, 138, "gNavyHdr", "prepare & check_isolation", "#FFFFFF", "#CBD5E1", [
        ("• SHA-256 input & schema lock", 17.2, "#334155", False, 28),
        ("• Zero-prompt 4-GPU defaults", 17.2, "#334155", False, 28),
        ("• Builds clean task/ workspace", 17.2, "#1D4ED8", True, 28),
    ], hdr_h=40))
    b.append(card(36, 232, 314, 126, "gBlueHdr", "Public task/ Directory", "#EFF6FF", "#93C5FD", [
        ("• description.md + sample CSV", 17.2, "#1E293B", False, 27),
        ("• train.csv / val.csv / images/", 17.2, "#1E293B", False, 27),
        ("• Unlabeled test features only", 17.2, "#1D4ED8", True, 27),
    ], hdr_h=38))
    b.append('<rect x="36" y="378" width="314" height="32" rx="6" fill="#FEE2E2" stroke="#DC2626" stroke-width="1.8"/>')
    b.append('<text x="193" y="400" text-anchor="middle" fill="#991B1B" font-size="14.5" font-weight="bold">STRICT LABEL FIREWALL (No Core Access)</text>')
    b.append(card(36, 424, 314, 126, "gRedHdr", "Sealed private/ Labels", "#FEF2F2", "#FCA5A5", [
        ("• private/test_labels.csv", 18, "#7F1D1D", True, 28),
        ("• Held strictly outside task_dir", 17, "#334155", False, 26),
    ], hdr_h=38))

    b.append('<rect x="396" y="18" width="1048" height="532" rx="16" fill="#EFF6FF" stroke="#2563EB" stroke-width="2.4"/>')
    b.append('<text x="920" y="48" text-anchor="middle" fill="#112B5C" font-size="22" font-weight="bold">2. SuperAgent Core 1.1 Search Engine (4 GPUs · 4 Workers)</text>')

    b.append(card(418, 68, 305, 138, "gBlueHdr", "Profiler & Method Catalog", "#FFFFFF", "#93C5FD", [
        ("• Auto modality detection", 17.2, "#334155", False, 27),
        ("• Offline SHA-256 + forward probe", 16.5, "#334155", False, 27),
        ("• Orthogonal draft hypotheses", 17.2, "#1D4ED8", True, 27),
    ], hdr_h=38))
    b.append(card(766, 68, 320, 138, "gNavyHdr", "AdaptiveController", "#FFFFFF", "#64748B", [
        ("• UCT &lt;-&gt; Elite tree scheduler", 17.2, "#334155", False, 27),
        ("• Value-density + complementarity", 16.8, "#334155", False, 27),
        ("• Decoupled slot_share timeout", 17.2, "#0F766E", True, 27),
    ], hdr_h=38))
    b.append(card(1126, 68, 296, 138, "gPurpleHdr", "Hybrid ExperienceStore", "#FAF5FF", "#C4B5FD", [
        ("• Qwen3-Embedding-0.6B (CPU)", 16.8, "#334155", False, 27),
        ("• CamelCase/CJK BM25 + RRF", 16.8, "#334155", False, 27),
        ("• +1 Gains & -1 Counterexamples", 16.8, "#5B21B6", True, 27),
    ], hdr_h=38))

    b.append(card(418, 238, 485, 142, "gTealHdr", "7 Operators & AST-Scoped Patching", "#FFFFFF", "#5EEAD4", [
        ("• draft · refine · repair · retrospect · borrow · synthesis · promote", 14.3, "#0F766E", True, 28),
        ("• Structured Planner (autonomous target) -> AST Patch", 16.3, "#334155", False, 27),
        ("• Role-stratified thinking & isolated worker execution", 16.3, "#334155", False, 27),
    ], hdr_h=38))
    b.append(card(938, 238, 484, 142, "gGoldHdr", "Risk-Gated Review & GPU Executor", "#FFFFFF", "#FCD34D", [
        ("• AST/Risk Reviewer blocks split & val-snooping leaks", 16.5, "#334155", False, 27),
        ("• 200-row SUPERAGENT_SMOKE=1 fast gate (&lt;=180s)", 16.5, "#334155", False, 27),
        ("• Dedicated per-slot GPU run + portable ./out/model/", 16.5, "#92400E", True, 27),
    ], hdr_h=38))

    b.append(card(418, 408, 1004, 126, "gNavyHdr", "Independent Assessment & Validated Final Builder (final.py)", "#FFFFFF", "#1E3A8A", [
        ("• Task-hint + exact-hit (1e-4) metric identification  |  Jump-triggered leak audit  |  Top-K model retention", 17.0, "#334155", False, 29),
        ("• stratified_halves Two-Half OOS Gate: Odd-pair greedy voting + Probability Sidecars (prob_argmax + log_loss)", 16.8, "#1D4ED8", True, 28),
    ], hdr_h=38))

    b.append('<rect x="1472" y="18" width="350" height="560" rx="16" fill="#F0FDFA" stroke="#0F766E" stroke-width="2.2"/>')
    b.append('<text x="1647" y="48" text-anchor="middle" fill="#0F766E" font-size="21" font-weight="bold">3. Verified Delivery</text>')
    b.append(card(1490, 68, 314, 210, "gTealHdr", "core11.py Offline Audit", "#FFFFFF", "#5EEAD4", [
        ("• Zero Core code execution", 17.2, "#334155", False, 29),
        ("• Verifies model symlink lineage", 17.2, "#334155", False, 29),
        ("• Replays vote / prob_argmax", 17.2, "#334155", False, 29),
        ("• Byte-exact combine.py check", 17.2, "#0F766E", True, 29),
    ], hdr_h=40))
    b.append(card(1490, 326, 314, 216, "gNavyHdr", "Final Evaluation & Report", "#FFFFFF", "#94A3B8", [
        ("• One-shot grading vs private/", 17.2, "#334155", False, 29),
        ("• Writes adapter/outcome.json", 17.2, "#334155", False, 29),
        ("• Lineage report & trace", 17.2, "#334155", False, 29),
        ("• Self-contained delivery/ bundle", 17.2, "#112B5C", True, 29),
    ], hdr_h=40))

    arrows = [
        (350, 295, 414, 150, "arrBlue"),
        (723, 137, 762, 137, "arrSlate"),
        (1086, 137, 1122, 137, "arrSlate"),
        (920, 206, 660, 234, "arrBlue"),
        (903, 309, 934, 309, "arrTeal"),
        (1180, 380, 1050, 404, "arrGold"),
        (1422, 470, 1486, 190, "arrTeal"),
        (1647, 278, 1647, 322, "arrTeal"),
    ]
    for x1, y1, x2, y2, mk in arrows:
        col = {"arrBlue": "#1D4ED8", "arrSlate": "#475569", "arrTeal": "#0F766E", "arrGold": "#B45309"}[mk]
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="2.8" marker-end="url(#{mk})"/>')
    b.append('<path d="M 350 536 L 382 536 L 382 568 L 1456 568 L 1456 490 L 1486 490" fill="none" stroke="#DC2626" stroke-width="2.8" stroke-dasharray="8,5" marker-end="url(#arrRed)"/>')
    b.append('<text x="920" y="588" text-anchor="middle" fill="#991B1B" font-size="14.5" font-weight="bold">Sealed private/ labels bypass Core 1.1 completely — accessed ONLY during post-run evaluation</text>')

    render_svg("svg_02_architecture", W, H, "\n".join(b))


# ==============================================================================
# 3. Slide 4: Search Engine I — Multi-Branch Tree, UCT <-> Elite & Chain Control
# ==============================================================================
def make_svg_03_search_tree():
    W, H = 1840, 530
    b = []
    # Left panel: Search Tree + Dual Selection
    b.append('<rect x="18" y="14" width="1068" height="502" rx="16" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="2"/>')
    b.append('<text x="552" y="46" text-anchor="middle" fill="#112B5C" font-size="21.5" font-weight="bold">Multi-Branch Search Tree &amp; Dual Explore–Exploit Selection (selection.py)</text>')

    # Root node
    b.append('<rect x="372" y="64" width="360" height="46" rx="23" fill="#112B5C"/>')
    b.append('<text x="552" y="93" text-anchor="middle" fill="#FFFFFF" font-size="18.5" font-weight="bold">Root (initial_drafts=3, max_drafts=5)</text>')

    # 3 Orthogonal branches
    branches = [
        (46, "Branch 1 · Draft 0", "TabICL / GBDT / ConvNeXt", "#1D4ED8", "#EFF6FF"),
        (392, "Branch 2 · Draft 1", "Emb-MLP / FT-Trans / DINOv2", "#0F766E", "#F0FDFA"),
        (738, "Branch 3 · Draft 2", "CatBoost+ET / EffNetV2", "#6D28D9", "#FAF5FF"),
    ]
    for bx, btitle, bsub, col, bg in branches:
        b.append(f'<line x1="552" y1="110" x2="{bx+160}" y2="146" stroke="{col}" stroke-width="2.6" marker-end="url(#arrSlate)"/>')
        b.append(f'<rect x="{bx}" y="150" width="320" height="68" rx="12" fill="{bg}" stroke="{col}" stroke-width="2"/>')
        b.append(f'<text x="{bx+160}" y="178" text-anchor="middle" fill="{col}" font-size="18" font-weight="bold">{btitle}</text>')
        b.append(f'<text x="{bx+160}" y="204" text-anchor="middle" fill="#334155" font-size="15.8">{bsub}</text>')

    # Dual Selection cards: UCT vs Elite
    b.append(card(42, 248, 495, 166, "gBlueHdr", "Exploration: tree_walk (Decaying UCT)", "#FFFFFF", "#93C5FD", [
        ("UCB(c) = R_c/N_c + c(t) * sqrt(ln N_p / N_c)  [unvisited = inf]", 15.8, "#1D4ED8", True, 29),
        ("• c(t) decays: 1.414 -> 0.50 (decay 0.01/step, ratios [0.3, 0.7])", 15.8, "#1E293B", False, 29),
        ("• Depth caps: max_refinements=3/node, max_repairs=1/node", 15.8, "#334155", False, 28),
    ], hdr_h=38, hdr_fs=18.5))

    b.append(card(566, 248, 496, 166, "gTealHdr", "Exploitation: elite_pick (Diversified Pool)", "#FFFFFF", "#5EEAD4", [
        ("• Top Pool: early_k=5 (3/branch) -> late_k=3 (2/branch)", 15.0, "#0F766E", True, 29),
        ("• Inverse-rank sampling w_i ~ 1/rank_i (elite_max_children=10)", 15.2, "#1E293B", False, 29),
        ("• chain_soft_switch + thread-local Dispatch.via_elite (race-free)", 15.0, "#334155", False, 28),
    ], hdr_h=38, hdr_fs=18.5))

    b.append('<line x1="206" y1="218" x2="260" y2="244" stroke="#1D4ED8" stroke-width="2.4" marker-end="url(#arrBlue)"/>')
    b.append('<line x1="552" y1="218" x2="430" y2="244" stroke="#1D4ED8" stroke-width="2.2" marker-end="url(#arrBlue)"/>')
    b.append('<line x1="552" y1="218" x2="690" y2="244" stroke="#0F766E" stroke-width="2.2" marker-end="url(#arrTeal)"/>')
    b.append('<line x1="898" y1="218" x2="840" y2="244" stroke="#0F766E" stroke-width="2.4" marker-end="url(#arrTeal)"/>')

    # Progress schedule bar at bottom of left panel
    b.append('<rect x="42" y="428" width="1020" height="72" rx="12" fill="#FFFFFF" stroke="#94A3B8" stroke-width="1.6"/>')
    b.append('<text x="58" y="454" fill="#112B5C" font-size="16" font-weight="bold">explore_weight(p) Schedule:</text>')
    # 3 segments
    b.append('<rect x="58" y="464" width="465" height="24" rx="5" fill="#1D4ED8"/>')
    b.append('<text x="290" y="481" text-anchor="middle" fill="#FFFFFF" font-size="13.8" font-weight="bold">p &lt; 0.5 :  w = 1.0 (Full UCT Exploration)</text>')
    b.append('<rect x="528" y="464" width="215" height="24" rx="5" fill="#7C3AED"/>')
    b.append('<text x="635" y="481" text-anchor="middle" fill="#FFFFFF" font-size="13.2" font-weight="bold">0.5..0.7 : Linear 1.0 -> 0.2</text>')
    b.append('<rect x="748" y="464" width="296" height="24" rx="5" fill="#0F766E"/>')
    b.append('<text x="896" y="481" text-anchor="middle" fill="#FFFFFF" font-size="13.8" font-weight="bold">p &gt;= 0.7 : w = 0.2 (80% Elite Focus)</text>')

    # Right panel: Chain Control, Smooth Reward & Concurrency-Safe Plateau
    b.append('<rect x="1108" y="14" width="714" height="502" rx="16" fill="#FFFFFF" stroke="#112B5C" stroke-width="2.4"/>')
    b.append('<path d="M 1108 30 Q 1108 14 1124 14 L 1806 14 Q 1822 14 1822 30 L 1822 62 L 1108 62 Z" fill="url(#gNavyHdr)"/>')
    b.append('<text x="1465" y="45" text-anchor="middle" fill="#FFFFFF" font-size="20.5" font-weight="bold">Chain Control, Smooth Reward &amp; Anti-Race (chain.py)</text>')

    b.append(card(1132, 80, 666, 198, "gPurpleHdr", "Smooth Step Reward &amp; Per-Node Chain State", "#FAF5FF", "#C4B5FD", [
        ("• Fail: -1.0  |  First branch ok: +0.5  |  Gain &lt; 1e-4: 0.0", 16.6, "#1E293B", False, 32),
        ("• New branch best: min(1.5, 0.75 + g / max(|s_prev|, 0.05))", 16.8, "#5B21B6", True, 32),
        ("• Per-node local no_gain_count (zero cross-worker race!):", 16.5, "#1E293B", True, 32),
        ("  resets to 0 on beat; +1 otherwise; dead_end at max_no_gain=3", 15.8, "#334155", False, 30),
        ("• Repair bounds: return root at depth=3, dead_end at depth=20", 15.8, "#334155", False, 28),
    ], hdr_h=38, hdr_fs=18.5))

    b.append(card(1132, 298, 666, 200, "gGoldHdr", "Forced Return &amp; Concurrency-Safe Plateau Windows", "#FFFBEB", "#FCD34D", [
        ("• forced_return prevents single-chain runaway:", 16.8, "#92400E", True, 32),
        ("  p>=0.4 every mid_every=3 steps | p>=0.8 w/ late_prob=0.5", 15.8, "#1E293B", False, 32),
        ("• Exempts nodes beating/tying best of recent_window=4", 16.2, "#0F766E", True, 32),
        ("• Anti-race plateau filter _after_best_finished(n, best):", 16.5, "#1D4ED8", True, 30),
        ("  counts only n.created_at >= best.finished_at (branch=3 / elite=6)", 15.4, "#334155", False, 28),
    ], hdr_h=38, hdr_fs=18.5))

    render_svg("svg_03_search_tree", W, H, "\n".join(b))


# ==============================================================================
# 4. Slide 5: Search Engine II — 7 Evolutionary Operators & Adaptive Admission
# ==============================================================================
def make_svg_04_operators_adaptive():
    W, H = 1840, 530
    b = []
    # Left panel: 7 Specialized Operators
    b.append('<rect x="18" y="14" width="1044" height="502" rx="16" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="2"/>')
    b.append('<text x="540" y="46" text-anchor="middle" fill="#112B5C" font-size="21.5" font-weight="bold">7 Specialized Evolutionary Operators (operators.py &amp; scheduling.py)</text>')

    # Row 1: Root & Local Evolution
    b.append('<text x="42" y="78" fill="#475569" font-size="15.5" font-weight="bold">Tier 1 · Root Seeding &amp; Local Evolution</text>')
    r1 = [
        (42, 92, 316, "1. draft (t = 0)", "assign_draft_hypotheses() seeds", "orthogonal families (parallel=4)", "#1D4ED8", "#EFF6FF"),
        (382, 92, 316, "2. refine", "Structured plan -> AST import-aware", "SEARCH/REPLACE targeted patch", "#0F766E", "#F0FDFA"),
        (722, 92, 316, "3. repair", "Traceback + failure recall fix;", "1 tier-2 retry for failed drafts", "#B45309", "#FFFBEB"),
    ]
    for x, y, w, title, l1, l2, col, bg in r1:
        b.append(f'<rect x="{x}" y="{y}" width="{w}" height="106" rx="12" fill="{bg}" stroke="{col}" stroke-width="2"/>')
        b.append(f'<text x="{x+w/2}" y="{y+30}" text-anchor="middle" fill="{col}" font-size="18.5" font-weight="bold">{title}</text>')
        b.append(f'<text x="{x+w/2}" y="{y+59}" text-anchor="middle" fill="#1E293B" font-size="15.2">{l1}</text>')
        b.append(f'<text x="{x+w/2}" y="{y+84}" text-anchor="middle" fill="#334155" font-size="15.2">{l2}</text>')

    # Row 2: Trajectory & Zero-LLM Checkpoint Continuation
    b.append('<text x="42" y="230" fill="#475569" font-size="15.5" font-weight="bold">Tier 2 · Trajectory Escalation &amp; Zero-LLM Compute Continuation</text>')
    b.append('<rect x="42" y="244" width="484" height="106" rx="12" fill="#FAF5FF" stroke="#6D28D9" stroke-width="2"/>')
    b.append('<text x="284" y="274" text-anchor="middle" fill="#6D28D9" font-size="18.5" font-weight="bold">4. retrospect (Plateau &amp; Trajectory)</text>')
    b.append('<text x="284" y="303" text-anchor="middle" fill="#1E293B" font-size="15.5">Reviews up to 10-step ancestor trajectory</text>')
    b.append('<text x="284" y="328" text-anchor="middle" fill="#334155" font-size="15.5">+ Level-2/3 escalation (ok_since&gt;=2 or all&gt;=5)</text>')

    b.append('<rect x="554" y="244" width="484" height="106" rx="12" fill="#F0FDFA" stroke="#0F766E" stroke-width="2.2"/>')
    b.append('<text x="796" y="274" text-anchor="middle" fill="#0F766E" font-size="18.5" font-weight="bold">5. promote (Zero-LLM Rung Continuation)</text>')
    b.append('<text x="796" y="303" text-anchor="middle" fill="#1E293B" font-size="15.5">Successive Halving: resumes saved checkpoint (max_rung=3)</text>')
    b.append('<text x="796" y="328" text-anchor="middle" fill="#0F766E" font-size="15.5" font-weight="bold">Verified by optimizer_steps + weight SHA-256 (0 LLM calls)</text>')

    # Row 3: Cross-Branch Crossover & Global Synthesis
    b.append('<text x="42" y="382" fill="#475569" font-size="15.5" font-weight="bold">Tier 3 · Cross-Branch Crossover &amp; Global Plateau Fusion (p &gt;= 0.5)</text>')
    b.append('<rect x="42" y="396" width="484" height="106" rx="12" fill="#EFF6FF" stroke="#1D4ED8" stroke-width="2"/>')
    b.append('<text x="284" y="426" text-anchor="middle" fill="#1D4ED8" font-size="18.5" font-weight="bold">6. borrow (Cross-Branch Grafting)</text>')
    b.append('<text x="284" y="455" text-anchor="middle" fill="#1E293B" font-size="15.5">Past p=0.5 (prob=0.3): transfers donor branch</text>')
    b.append('<text x="284" y="480" text-anchor="middle" fill="#334155" font-size="15.5">cards + code excerpt onto stalled branch (up to 5 refs)</text>')

    b.append('<rect x="554" y="396" width="484" height="106" rx="12" fill="#FFFBEB" stroke="#B45309" stroke-width="2"/>')
    b.append('<text x="796" y="426" text-anchor="middle" fill="#92400E" font-size="18.5" font-weight="bold">7. synthesis (Multi-Branch Fusion)</text>')
    b.append('<text x="796" y="455" text-anchor="middle" fill="#1E293B" font-size="15.5">Window [0.5, 0.83], global_window=4, &gt;=2 rich branches</text>')
    b.append('<text x="796" y="480" text-anchor="middle" fill="#334155" font-size="15.5">Recombines diverse branch reps from Root (max 2x)</text>')

    # Right panel: Adaptive Value-Density Admission
    b.append('<rect x="1086" y="14" width="736" height="502" rx="16" fill="#FFFFFF" stroke="#112B5C" stroke-width="2.4"/>')
    b.append('<path d="M 1086 30 Q 1086 14 1102 14 L 1806 14 Q 1822 14 1822 30 L 1822 62 L 1086 62 Z" fill="url(#gNavyHdr)"/>')
    b.append('<text x="1454" y="45" text-anchor="middle" fill="#FFFFFF" font-size="20.5" font-weight="bold">Adaptive Value-Density Admission (control/adaptive.py)</text>')

    b.append(card(1110, 80, 688, 212, "gBlueHdr", "Complementarity-Guided Priority &amp; Multi-Slot Ranking", "#EFF6FF", "#93C5FD", [
        ("priority = merit * p_ok / max(need_secs, 1)", 17.5, "#1D4ED8", True, 32),
        ("merit = 0.2 + 1/(1+0.12*r) + 0.4/(1+children) + min(0.5, comp)", 15.8, "#112B5C", True, 32),
        ("• comp = mean((best != y_val) &amp; (node == y_val)) [diversity!]", 15.8, "#0F766E", True, 30),
        ("• rank_proposals(): multi-slot proposal keys (slot=k)", 15.8, "#1E293B", False, 29),
        ("• Autonomous target=None -> planner picks data_prep/model_def/train_eval", 15.0, "#334155", False, 28),
    ], hdr_h=38, hdr_fs=18.2))

    b.append(card(1110, 310, 688, 188, "gTealHdr", "Decoupled Slot-Scaled Execution Timeout &amp; Quota Safety", "#F0FDFA", "#5EEAD4", [
        ("• Priority estimates exclude crashed runs: median(ok_script_secs)", 15.8, "#1E293B", False, 32),
        ("• Kill ceiling: max(2.5 * q0.8(ok_secs), 0.75 * slot_share)", 16.5, "#0F766E", True, 32),
        ("  where slot_share = (T_rem - T_res) * num_gpus / rem_nodes", 15.6, "#1E293B", False, 30),
        ("• Auto-scales cleanly: ~40s (smoke) -> 300s+ (1h) -> 1000s+ (6h)", 15.8, "#1D4ED8", True, 30),
    ], hdr_h=38, hdr_fs=18.2))

    render_svg("svg_04_operators_adaptive", W, H, "\n".join(b))


# ==============================================================================
# 5. Slide 6: Hybrid RRF Experience Knowledge Base & Method Priors
# ==============================================================================
def make_svg_05_hybrid_memory():
    W, H = 1840, 520
    b = []
    b.append(card(20, 26, 360, 200, "gNavyHdr", "Committed Run Records", "#F8FAFC", "#94A3B8", [
        ("• Plan &amp; Method Summary", 17.5, "#334155", False, 32),
        ("• Validation Gain: effect in {+1, -1}", 17.5, "#1D4ED8", True, 32),
        ("• Diagnostic Traceback / Timeout", 17.5, "#334155", False, 32),
        ("• Persisted in experience/records.json", 16.2, "#64748B", False, 28),
    ], hdr_h=42))

    b.append(card(20, 260, 360, 205, "gTealHdr", "Static Method Priors", "#F0FDFA", "#5EEAD4", [
        ("• method_catalog.yaml (by modality)", 17.5, "#1E293B", False, 32),
        ("• SHA-256 + forward_ok local cache", 17.5, "#0F766E", True, 32),
        ("• Task cards (counterexamples)", 17.5, "#1E293B", False, 32),
        ("• Orthogonal Draft 0/1/2 seeding", 17.5, "#1D4ED8", True, 30),
    ], hdr_h=42))

    b.append(card(445, 22, 470, 200, "gPurpleHdr", "Dense Semantic Index (Local CPU)", "#FAF5FF", "#C4B5FD", [
        ("• Model: Qwen/Qwen3-Embedding-0.6B", 18.2, "#5B21B6", True, 33),
        ("• 1024-d L2-normalized vectors (max_len=1024)", 17.0, "#334155", False, 31),
        ("• Cosine floor >= 0.35  |  Zero GPU contention", 17.2, "#0F766E", True, 31),
        ("• Captures cross-phrasing ML concepts", 17.2, "#334155", False, 30),
    ], hdr_h=42))

    b.append(card(445, 258, 470, 204, "gBlueHdr", "Sparse Lexical Index (Okapi BM25)", "#EFF6FF", "#93C5FD", [
        ("• CamelCase + CJK Tokenizer (k1=1.5, b=0.75)", 17.5, "#1D4ED8", True, 33),
        ("• StratifiedKFold -> [stratifiedkfold, stratified, kfold]", 15.6, "#334155", False, 31),
        ("• Exact match on library &amp; exception symbols", 16.8, "#1E293B", True, 31),
        ("  (e.g., LGBMClassifier, ConvNeXt, CUDA OOM)", 16.5, "#475569", False, 30),
    ], hdr_h=42))

    b.append(card(980, 118, 390, 250, "gGoldHdr", "Weighted RRF Rank Fusion", "#FFFBEB", "#FCD34D", [
        ("Reciprocal Rank Fusion:", 18.2, "#92400E", True, 34),
        ("score(i) = 0.5 / (60 + r_kw)", 18.8, "#112B5C", True, 32),
        ("         + 0.5 / (60 + r_vec)", 18.8, "#112B5C", True, 36),
        ("• Scale-free rank combination", 17.2, "#334155", False, 30),
        ("• Graceful keyword fallback", 17.2, "#334155", False, 30),
    ], hdr_h=42))

    b.append(card(1435, 38, 385, 424, "gNavyHdr", "Evidence-Guided Planner", "#FFFFFF", "#1E3A8A", [
        ("Dual Contrastive Context:", 18.8, "#112B5C", True, 36),
        ("✓ Top-2 Positive Records (+1)", 18.2, "#0F766E", True, 28),
        ("    Proven gains to build upon", 16.8, "#475569", False, 36),
        ("✗ Top-2 Negative Records (-1)", 18.2, "#DC2626", True, 28),
        ("    Prioritizes direct failed children", 16.5, "#475569", False, 38),
        ("Outputs Experiment Plan:", 18.2, "#1D4ED8", True, 30),
        ("• Autonomous component target", 17.0, "#334155", False, 28),
        ("• AST-verified symbol + imports", 17.0, "#334155", False, 28),
    ], hdr_h=44))

    arrows = [
        (380, 125, 441, 125, "arrBlue"),
        (380, 155, 441, 330, "arrBlue"),
        (915, 130, 976, 210, "arrGold"),
        (915, 360, 976, 275, "arrGold"),
        (1370, 245, 1431, 245, "arrBlue"),
    ]
    for x1, y1, x2, y2, mk in arrows:
        col = {"arrBlue": "#1D4ED8", "arrGold": "#B45309"}[mk]
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="2.8" marker-end="url(#{mk})"/>')
    b.append('<path d="M 200 465 L 200 494 L 1627 494 L 1627 466" fill="none" stroke="#0F766E" stroke-width="2.8" stroke-dasharray="7,5" marker-end="url(#arrTeal)"/>')
    b.append('<text x="915" y="513" text-anchor="middle" fill="#0F766E" font-size="14.5" font-weight="bold">Static Method Catalog &amp; Task Counterexamples directly condition Draft &amp; Planner prompts</text>')

    render_svg("svg_05_hybrid_memory", W, H, "\n".join(b))


# ==============================================================================
# 6. Slide 7: Two-Half Out-of-Sample Ensemble Gate & Probability Sidecars
# ==============================================================================
def make_svg_06_validated_ensemble():
    W, H = 1840, 490
    b = []
    b.append(card(20, 35, 395, 420, "gNavyHdr", "1. Top-6 Diverse Candidates", "#F8FAFC", "#94A3B8", [
        ("Per-Branch Diversity Cap:", 18.2, "#112B5C", True, 32),
        ("• Max top_per_branch = 5", 17.2, "#334155", False, 30),
        ("• Shared validation IDs >= 90%", 17.2, "#334155", False, 36),
        ("Dual Output Channels:", 18.2, "#1D4ED8", True, 32),
        ("• predictions.csv (contract labels)", 17.2, "#334155", False, 30),
        ("• val/test_probabilities.csv", 17.8, "#0F766E", True, 28),
        ("  (calibrated K-class sidecars)", 16.8, "#475569", False, 30),
    ], hdr_h=42))

    b.append(card(465, 35, 415, 420, "gBlueHdr", "2. Metric &amp; Candidate Plans", "#EFF6FF", "#93C5FD", [
        ("Exact Metric Disambiguation:", 18.2, "#1D4ED8", True, 32),
        ("• Task-desc hint + 1e-4 exact_hits", 17.2, "#334155", False, 30),
        ("• Eliminates precision_macro drift", 17.2, "#0F766E", True, 36),
        ("Evaluated Ensemble Plans:", 18.2, "#112B5C", True, 32),
        ("• Discrete: top2/3/4/6 + pair-greedy", 17.2, "#334155", False, 30),
        ("• Sidecar: prob_topK + prob_greedy", 17.8, "#5B21B6", True, 28),
        ("  (weighted prob sum -> argmax)", 16.8, "#475569", False, 30),
    ], hdr_h=42))

    b.append(card(930, 35, 455, 420, "gGoldHdr", "3. stratified_halves OOS Gate", "#FFFBEB", "#FCD34D", [
        ("Stratified Val Split -> Half A &amp; B:", 18.2, "#92400E", True, 34),
        ("• Fit weights on A  -->  Score on B", 17.8, "#1E293B", True, 30),
        ("• Fit weights on B  -->  Score on A", 17.8, "#1E293B", True, 36),
        ("Strict Acceptance Criterion:", 18.2, "#112B5C", True, 32),
        ("• Must win BOTH halves by min_gain", 17.2, "#334155", False, 30),
        ("• Or tie metric &amp; beat single by", 17.2, "#334155", False, 28),
        ("  >0.001 cross-entropy log_loss!", 17.8, "#0F766E", True, 30),
    ], hdr_h=42))

    b.append(card(1435, 35, 385, 420, "gTealHdr", "4. Verified Primary final/", "#F0FDFA", "#5EEAD4", [
        ("If Accepted -> Ensemble:", 18.2, "#0F766E", True, 32),
        ("• Intel Image: top6 -> 95.33% test", 17.2, "#1E293B", True, 30),
        ("  (beats 95.13% best single!)", 16.8, "#475569", False, 36),
        ("If Rejected -> Safe Fallback:", 18.2, "#112B5C", True, 32),
        ("• Keeps best single (zero harm)", 17.2, "#334155", False, 34),
        ("Deterministic Replay:", 18.2, "#1D4ED8", True, 30),
        ("• Byte-exact combine.py &amp; core11", 17.2, "#334155", False, 28),
    ], hdr_h=42))

    for x1, x2, mk, col in [(415, 461, "arrBlue", "#1D4ED8"), (880, 926, "arrGold", "#B45309"), (1385, 1431, "arrTeal", "#0F766E")]:
        b.append(f'<line x1="{x1}" y1="245" x2="{x2}" y2="245" stroke="{col}" stroke-width="3.2" marker-end="url(#{mk})"/>')

    render_svg("svg_06_validated_ensemble", W, H, "\n".join(b))


# ==============================================================================
# 7. Slide 11: Task-Agnostic Generalizable Hardening (541 Passing Tests)
# ==============================================================================
def make_svg_07_generalization():
    W, H = 1840, 470
    pillars = [
        ("gBlueHdr", "#EFF6FF", "#93C5FD", "1. Concurrency-Safe Tree Search",
         ["• Per-node local no_gain_count",
          "• _after_best_finished plateau filter",
          "• Multi-slot rank_proposals + quota",
          "• Thread-local Dispatch.via_elite",
          "Scope: Parallel multi-GPU search"]),
        ("gPurpleHdr", "#FAF5FF", "#C4B5FD", "2. AST Patching &amp; Smart Recall",
         ["• AST import-aware SEARCH/REPLACE",
          "• Autonomous target (data/model/train)",
          "• CamelCase + CJK Okapi BM25 tokens",
          "• Direct failed-child (-1) prioritization",
          "Scope: Tabular, Vision, Text, Multi."]),
        ("gTealHdr", "#F0FDFA", "#5EEAD4", "3. Stratified OOS &amp; Sidecar Gate",
         ["• stratified_halves class-balanced CV",
          "• val/test_probabilities.csv sidecars",
          "• +2 pair greedy escapes 1:1 vote lock",
          "• Anti-val-snooping AST/risk checks",
          "Scope: Binary &amp; K-class tasks"]),
        ("gGoldHdr", "#FFFBEB", "#FCD34D", "4. Orthogonal Seed &amp; Slot Scaling",
         ["• assign_draft_hypotheses() at t=0",
          "• Decoupled slot_share kill ceiling",
          "• Zero-LLM promote rung continuation",
          "• 541 offline tests verified cleanly",
          "Scope: Any wall-clock &amp; GPU setup"]),
    ]
    b = []
    cw, gap, x0 = 430, 26, 20
    for idx, (grad, bg, stroke, title, lines) in enumerate(pillars):
        x = x0 + idx * (cw + gap)
        spec_lines = []
        for j, ln in enumerate(lines):
            is_last = j == len(lines) - 1
            spec_lines.append((ln, 17.5 if not is_last else 17.2, "#112B5C" if is_last else "#1E293B", is_last, 42 if j == 3 else 36))
        b.append(card(x, 20, cw, 430, grad, title, bg, stroke, spec_lines, hdr_h=46, hdr_fs=18.5))
    render_svg("svg_07_generalization", W, H, "\n".join(b))


if __name__ == "__main__":
    make_svg_01_failure_modes()
    make_svg_02_architecture()
    make_svg_03_search_tree()
    make_svg_04_operators_adaptive()
    make_svg_05_hybrid_memory()
    make_svg_06_validated_ensemble()
    make_svg_07_generalization()
