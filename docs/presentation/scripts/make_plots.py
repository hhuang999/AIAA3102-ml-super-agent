"""Generate 4 crisp 300-DPI experimental charts from verified SuperAgent & MLEvolve runs."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "assets"
A.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.1,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

NAVY = "#112B5C"
BLUE = "#1D4ED8"
TEAL = "#0F766E"
GOLD = "#D97706"
RED = "#DC2626"
PURPLE = "#6D28D9"
SLATE = "#64748B"


# ==============================================================================
# Plot 1: Token Spend Concentration & 68.8% Reasoning Token Reduction (Backup B3)
# ==============================================================================
def make_plot_01():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 3.7), dpi=300)
    fig.patch.set_facecolor("white")

    tasks = ["Airline Tabular\n(¥18.12 enforced)", "Intel Image 6-Class\n(¥18.16 enforced)"]
    reasoning = np.array([52.1, 53.1])
    uncached = np.array([29.0, 28.9])
    visible = np.array([12.0, 12.4])
    cached = np.array([7.0, 5.6])

    x = np.arange(len(tasks))
    w = 0.48
    ax1.bar(x, reasoning, w, label="Reasoning (Thinking) Output", color=RED, alpha=0.88)
    ax1.bar(x, uncached, w, bottom=reasoning, label="Uncached Input", color=BLUE, alpha=0.85)
    ax1.bar(x, visible, w, bottom=reasoning + uncached, label="Visible Code/JSON Output", color=TEAL, alpha=0.88)
    ax1.bar(x, cached, w, bottom=reasoning + uncached + visible, label="Cached Input", color="#94A3B8")

    for i in range(len(tasks)):
        ax1.text(i, reasoning[i] / 2, f"{reasoning[i]:.1f}%\n(¥{[9.44, 9.65][i]:.2f})", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
        ax1.text(i, reasoning[i] + uncached[i] / 2, f"{uncached[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)
        ax1.text(i, reasoning[i] + uncached[i] + visible[i] / 2, f"{visible[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    ax1.set_ylim(0, 108)
    ax1.set_ylabel("Share of Enforced LLM Spend (%)", fontsize=10.5, fontweight="bold", color=NAVY)
    ax1.set_title("Cost Monitor Diagnosis: >52% Spent in Thinking", fontsize=11.5, fontweight="bold", color=NAVY, pad=10)
    ax1.set_xticks(x, tasks, fontsize=10, fontweight="bold")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=8.8)
    ax1.grid(axis="y")
    ax1.set_axisbelow(True)

    labels = ["SuperAgent v0\n(Unthrottled)", "MLEvolve\n(1h Baseline)", "SuperAgent v1\n(Role-Stratified)"]
    vals = [2.359, 0.937, 0.735]
    cols = [RED, SLATE, TEAL]
    bars = ax2.bar(np.arange(3), vals, width=0.5, color=cols, alpha=0.9)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.3f}M", ha="center", va="bottom", fontweight="bold", fontsize=10, color=NAVY)
    ax2.annotate("-68.8% CoT Tokens!\n(patch_coder CoT = 0)", xy=(2, 0.735), xytext=(1.18, 1.72),
                 arrowprops=dict(arrowstyle="->", color=TEAL, lw=2),
                 fontsize=9.8, fontweight="bold", color=TEAL,
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#F0FDFA", edgecolor=TEAL))
    ax2.set_ylim(0, 2.75)
    ax2.set_ylabel("Total Reasoning Tokens (Millions)", fontsize=10.5, fontweight="bold", color=NAVY)
    ax2.set_title("Effect of Role-Stratified Thinking & Risk Review", fontsize=11.5, fontweight="bold", color=NAVY, pad=10)
    ax2.set_xticks(np.arange(3), labels, fontsize=9.5, fontweight="bold")
    ax2.grid(axis="y")
    ax2.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(A / "plot_01_token_economics.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ==============================================================================
# Plot 2: Intel Image 6-Class Test F1 & Search Execution Outcomes Across Runs (Slide 8)
# ==============================================================================
def make_plot_02():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 3.6), dpi=300, gridspec_kw={"width_ratios": [1.28, 1.0]})
    fig.patch.set_facecolor("white")

    classes = ["forest", "sea", "street", "buildings", "mountain", "glacier"]
    f1s = [99.68, 98.73, 95.98, 95.47, 91.60, 91.24]
    y = np.arange(len(classes))
    bars = ax1.barh(y, f1s, height=0.58, color=BLUE, alpha=0.86)
    ax1.axvline(95.33, color=TEAL, linestyle="--", lw=2, label="Validated top6 Ensemble Acc: 95.33%")
    ax1.axvline(95.13, color=GOLD, linestyle=":", lw=2, label="Best Single Model Acc: 95.13%")
    for b, v in zip(bars, f1s):
        ax1.text(v + 0.25, b.get_y() + b.get_height() / 2, f"{v:.2f}%", va="center", fontsize=9.5, fontweight="bold", color=NAVY)
    ax1.set_xlim(86, 102.5)
    ax1.set_yticks(y, classes, fontsize=10, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlabel("Held-Out Private Test F1 / Accuracy (%)", fontsize=10, fontweight="bold", color=NAVY)
    ax1.set_title("Intel Image Classification (3,000 Sealed Test Images)", fontsize=11.3, fontweight="bold", color=NAVY, pad=8)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.7)
    ax1.grid(axis="x")
    ax1.set_axisbelow(True)

    # Right: Search Execution Outcomes Across Runs (5 Branches each)
    runs = ["Airline Tabular\n(v0 · 120 nodes)", "Intel Image\n(Vision · 85 nodes)", "Airline Tabular\n(v1 · 57 nodes)"]
    valid = np.array([78, 72, 49], dtype=float)
    timeout = np.array([31, 2, 1], dtype=float)
    other_err = np.array([11, 11, 7], dtype=float)
    totals = valid + timeout + other_err
    valid_pct = valid / totals * 100.0
    timeout_pct = timeout / totals * 100.0
    err_pct = other_err / totals * 100.0

    x = np.arange(3)
    w = 0.46
    ax2.bar(x, valid_pct, w, label="Valid Evaluated Nodes (%)", color=TEAL, alpha=0.9)
    ax2.bar(x, timeout_pct, w, bottom=valid_pct, label="Timeout Kills (%)", color=RED, alpha=0.86)
    ax2.bar(x, err_pct, w, bottom=valid_pct + timeout_pct, label="Runtime / Other Errors (%)", color=SLATE, alpha=0.75)

    for i in range(3):
        ax2.text(i, valid_pct[i] / 2, f"{int(valid[i])} ok\n({valid_pct[i]:.1f}%)", ha="center", va="center", color="white", fontweight="bold", fontsize=9.2)
        if timeout_pct[i] >= 10:
            ax2.text(i, valid_pct[i] + timeout_pct[i] / 2, f"{int(timeout[i])} TO\n({timeout_pct[i]:.1f}%)", ha="center", va="center", color="white", fontweight="bold", fontsize=8.8)

    ax2.annotate("11 zero-LLM\npromote nodes!", xy=(2, 86.0), xytext=(1.12, 95.5),
                 arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.6),
                 fontsize=8.6, fontweight="bold", color=NAVY,
                 bbox=dict(boxstyle="round,pad=0.28", facecolor="#EFF6FF", edgecolor=BLUE))

    ax2.set_ylim(0, 118)
    ax2.set_xticks(x, runs, fontsize=9.2, fontweight="bold")
    ax2.set_ylabel("Share of Search Tree Nodes (%)", fontsize=10, fontweight="bold", color=NAVY)
    ax2.set_title("Search Node Outcomes Across Modalities (5 Branches)", fontsize=11.0, fontweight="bold", color=NAVY, pad=8)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=8.3)
    ax2.grid(axis="y")
    ax2.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(A / "plot_02_multimodal_search.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ==============================================================================
# Plot 3: Breaking the Timeout Death Spiral & Fixing Val Drift (v0 -> v1) (Slide 9)
# ==============================================================================
def make_plot_03():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 3.55), dpi=300)
    fig.patch.set_facecolor("white")

    metrics = ["Node Success\nRate (%)", "Timeout Killed\nNodes (count)"]
    v0 = [65.0, 31.0]
    v1 = [86.0, 1.0]
    x = np.arange(2)
    w = 0.32
    b0 = ax1.bar(x - w / 2, v0, w, label="Pre-Fix SuperAgent (v0)", color=RED, alpha=0.85)
    b1 = ax1.bar(x + w / 2, v1, w, label="Upgraded SuperAgent (v1)", color=TEAL, alpha=0.9)
    for b in b0:
        v = b.get_height()
        lbl = f"{v:.1f}%" if b.get_x() < 0.5 else f"{int(v)} kills"
        ax1.text(b.get_x() + b.get_width() / 2, v + 1.8, lbl, ha="center", fontweight="bold", fontsize=9.8, color=RED)
    for b in b1:
        v = b.get_height()
        lbl = f"{v:.1f}%" if b.get_x() < 0.5 else f"{int(v)} kill"
        ax1.text(b.get_x() + b.get_width() / 2, v + 1.8, lbl, ha="center", fontweight="bold", fontsize=9.8, color=TEAL)
    ax1.set_ylim(0, 102)
    ax1.set_xticks(x, metrics, fontsize=10, fontweight="bold")
    ax1.set_title("Eliminating the 25-Second Timeout Spiral", fontsize=11.3, fontweight="bold", color=NAVY, pad=8)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9)
    ax1.grid(axis="y")
    ax1.set_axisbelow(True)

    groups = ["Pre-Fix v0\n(Mixed OOF vs val.csv)", "Upgraded v1\n(Aligned val.csv Rule)"]
    rep = [93.16, 92.94]
    act = [92.79, 92.94]
    x2 = np.arange(2)
    r0 = ax2.bar(x2 - w / 2, rep, w, label="Reported Node Score (%)", color=BLUE, alpha=0.85)
    r1 = ax2.bar(x2 + w / 2, act, w, label="True val.csv Accuracy (%)", color=NAVY, alpha=0.92)
    for b in list(r0) + list(r1):
        v = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.2f}%", ha="center", fontweight="bold", fontsize=9.5, color=NAVY)
    ax2.annotate("0.37% gap -> picked\nprecision_macro (rejected!)", xy=(0.16, 92.79), xytext=(0.05, 92.22),
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.8), fontsize=8.8, fontweight="bold", color=RED,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor=RED))
    ax2.annotate("0.00% gap -> exact\naccuracy & +0.15% val gain!", xy=(1.16, 92.94), xytext=(0.82, 92.22),
                 arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.8), fontsize=8.8, fontweight="bold", color=TEAL,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FDFA", edgecolor=TEAL))
    ax2.set_ylim(92.0, 93.45)
    ax2.set_xticks(x2, groups, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Validation Accuracy (%)", fontsize=10, fontweight="bold", color=NAVY)
    ax2.set_title("Validation Alignment & Course Contract Pass", fontsize=11.3, fontweight="bold", color=NAVY, pad=8)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.8)
    ax2.grid(axis="y")
    ax2.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(A / "plot_03_timeout_fix.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ==============================================================================
# Plot 4: Controlled 1-Hour Head-to-Head (SuperAgent v1 vs MLEvolve) (Slide 10)
# ==============================================================================
def make_plot_04():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 3.65), dpi=300)
    fig.patch.set_facecolor("white")

    # Panel 1: 1h Search Throughput & Valid Yield
    b_valid = ax1.bar([0, 1], [49, 27], width=0.46, color=[TEAL, SLATE], alpha=0.92, label="Valid Evaluated Nodes")
    ax1.bar([0, 1], [8, 3], bottom=[49, 27], width=0.46, color=["#93C5FD", "#CBD5E1"], label="Timeout / Failed")
    for b, lbl in zip(b_valid, ["57 total\n(49 ok,\n11 promote)", "30 total\n(27 ok)"]):
        v = b.get_height() + (8 if b.get_x() < 0.5 else 3)
        ax1.text(b.get_x() + b.get_width() / 2, v + 1.5, lbl, ha="center", va="bottom", fontweight="bold", fontsize=8.5, color=NAVY)
    ax1.annotate("1.9x throughput!", xy=(0.15, 57), xytext=(0.42, 63),
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.6),
                 fontsize=8.8, fontweight="bold", color=BLUE)
    ax1.set_ylim(0, 74)
    ax1.set_xticks([0, 1], ["SuperAgent\n(53.6 min)", "MLEvolve\n(60.0 min)"], fontsize=9.2, fontweight="bold")
    ax1.set_ylabel("Evaluated Nodes in 1h", fontsize=9.8, fontweight="bold", color=NAVY)
    ax1.set_title("1h Search Throughput & Yield", fontsize=10.5, fontweight="bold", color=NAVY)
    ax1.grid(axis="y")
    ax1.set_axisbelow(True)

    # Panel 2: Held-Out Private Test Accuracy (20,000 Rows)
    x = np.arange(2)
    w = 0.34
    sa = [92.175, 92.245]
    ml = [92.170, 92.235]
    r0 = ax2.bar(x - w / 2, sa, w, color=BLUE, alpha=0.92, label="SuperAgent")
    r1 = ax2.bar(x + w / 2, ml, w, color=SLATE, alpha=0.85, label="MLEvolve")
    for b in r0:
        v = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}%", ha="center", va="bottom", fontweight="bold", fontsize=8.1, color=BLUE, rotation=12)
    for b in r1:
        v = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}%", ha="center", va="bottom", fontweight="bold", fontsize=8.1, color="#334155", rotation=12)
    ax2.set_ylim(92.12, 92.285)
    ax2.set_xticks(x, ["Primary\nDelivered", "Best Single\nDiscovered"], fontsize=9.2, fontweight="bold")
    ax2.set_ylabel("Sealed Test Accuracy (%)", fontsize=9.8, fontweight="bold", color=NAVY)
    ax2.set_title("20k Test Accuracy Comparison", fontsize=10.5, fontweight="bold", color=NAVY)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.0)
    ax2.grid(axis="y")
    ax2.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(A / "plot_04_head2head.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    make_plot_01()
    make_plot_02()
    make_plot_03()
    make_plot_04()
    print("Generated all 4 300-DPI Matplotlib plots.")
