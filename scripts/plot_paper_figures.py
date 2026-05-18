"""
Generate remaining SE-RAG paper figures and tables.

Outputs in paper/:
  - fig_multi_gt_comparison.png/.pdf
  - fig_gt_inconsistency.png/.pdf
  - fig_main_results_table.png/.pdf
  - fig_per_action_table.png/.pdf
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper"

COLORS = {
    "primary": "#2E86AB",
    "accent": "#A23B72",
    "highlight": "#F18F01",
    "warm": "#C9A66B",
    "soft": "#E6D3A7",
    "text": "#2D3436",
    "muted": "#5F6B73",
    "grid": "#E0E0E0",
    "row_alt": "#F5F5F5",
    "se_rag_bg": "#E8F4F8",
    "green": "#27AE60",
    "red": "#E74C3C",
}

METHOD_COLORS = {
    "SE-RAG": COLORS["primary"],
    "bm25_rag": COLORS["highlight"],
    "no_rag": COLORS["accent"],
    "dense_rag": COLORS["warm"],
    "tfidf_rag": COLORS["soft"],
}

DISPLAY_NAMES = {
    "SE-RAG": "SE-RAG",
    "se_rag": "SE-RAG",
    "bm25_rag": "BM25 RAG",
    "no_rag": "No-RAG",
    "dense_rag": "Dense RAG",
    "tfidf_rag": "TF-IDF RAG",
}


def arrow(ax, start, end, color=None, lw=1.45, linestyle="-", arrowstyle="-|>", zorder=3):
    """Draw an arrow segment using FancyArrowPatch."""
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=arrowstyle,
        mutation_scale=10.5,
        linewidth=lw,
        linestyle=linestyle,
        color=color or COLORS["muted"],
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(patch)
    return patch


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def pct_formatter(x, _pos):
    return f"{int(x)}%"


def save_figure(fig, basename):
    png_path = OUT_DIR / f"{basename}.png"
    pdf_path = OUT_DIR / f"{basename}.pdf"
    plt.tight_layout()
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, dpi=300, facecolor="white")
    plt.close(fig)
    png_size = png_path.stat().st_size / 1024
    pdf_size = pdf_path.stat().st_size / 1024
    print(f"Saved {png_path.relative_to(ROOT)} ({png_size:.1f} KB)")
    print(f"Saved {pdf_path.relative_to(ROOT)} ({pdf_size:.1f} KB)")


def generate_multi_gt_comparison():
    methods = ["SE-RAG", "bm25_rag", "no_rag", "dense_rag", "tfidf_rag"]
    multi_gt = np.array([90.4, 86.9, 69.8, 76.6, 80.1])
    single = np.array([63.3, 54.9, 23.5, 30.4, 45.3])

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(methods))
    width = 0.34
    bar_colors = [METHOD_COLORS[m] for m in methods]

    multi_bars = ax.bar(
        x - width / 2,
        multi_gt,
        width,
        color=bar_colors,
        edgecolor="white",
        linewidth=1.2,
        label="Multi-GT accuracy",
        zorder=3,
    )
    single_bars = ax.bar(
        x + width / 2,
        single,
        width,
        color=bar_colors,
        alpha=0.38,
        hatch="///",
        edgecolor=COLORS["text"],
        linewidth=0.6,
        label="Single-turn accuracy",
        zorder=3,
    )

    baseline = 69.8
    ax.axhline(baseline, color=COLORS["muted"], linestyle=(0, (4, 3)), linewidth=1.15, zorder=2)
    ax.text(
        len(methods) - 0.05,
        baseline + 1.8,
        "No-RAG baseline (69.8%)",
        ha="right",
        va="bottom",
        fontsize=9,
        color=COLORS["muted"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
    )

    for bar, value in zip([multi_bars[0], single_bars[0]], [multi_gt[0], single[0]]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=COLORS["text"],
        )

    ax.set_title("Overall Performance Comparison (1000 Queries)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in methods], fontsize=9, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", color=COLORS["grid"], linestyle="--", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    metric_handles = [
        Patch(facecolor=COLORS["primary"], edgecolor="white", label="Multi-GT accuracy"),
        Patch(
            facecolor=COLORS["primary"],
            edgecolor=COLORS["text"],
            alpha=0.38,
            hatch="///",
            label="Single-turn accuracy",
        ),
    ]
    ax.legend(handles=metric_handles, loc="upper right", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "fig_multi_gt_comparison")


def generate_gt_inconsistency():
    rows = [
        ("driver_unavailable", 58.5),
        ("vehicle_breakdown", 55.6),
        ("order_cancel", 53.5),
        ("bridge_weight_limit", 52.0),
        ("order_modify", 49.7),
        ("road_closed", 49.4),
        ("traffic_congestion", 48.8),
        ("Average", 48.0),
    ]

    labels = [name.replace("_", " ").title() for name, _ in rows]
    values = np.array([value for _, value in rows])

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cmap = LinearSegmentedColormap.from_list("inconsistency_red", ["#FADBD8", "#922B21"])
    norm = Normalize(vmin=values.min(), vmax=values.max())
    colors = [cmap(norm(v)) for v in values]

    y = np.arange(len(labels))
    bars = ax.barh(y, values, height=0.58, color=colors, edgecolor="white", linewidth=1.0, zorder=3)

    average = 48.0
    ax.axvline(average, color=COLORS["muted"], linestyle=(0, (4, 3)), linewidth=1.15, zorder=2)
    ax.text(
        average + 0.8,
        -0.60,
        "Average (48.0%)",
        ha="left",
        va="center",
        fontsize=9,
        color=COLORS["muted"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
    )

    for bar, value in zip(bars, values):
        ax.text(
            value + 0.9,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=COLORS["text"],
        )

    ax.set_title("Ground Truth Inconsistency Across Event Types", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Inconsistency Rate", fontsize=11)
    ax.set_xlim(0, 70)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", color=COLORS["grid"], linestyle="--", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "fig_gt_inconsistency")


def style_table(table, n_rows, n_cols, se_rag_row=None):
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D9D9D9")
        cell.set_linewidth(0.8)
        cell.PAD = 0.05

        if row == 0:
            cell.set_facecolor(COLORS["text"])
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.get_text().set_fontsize(10)
            cell.get_text().set_ha("center")
            continue

        if se_rag_row is not None and row == se_rag_row:
            cell.set_facecolor(COLORS["se_rag_bg"])
        elif row % 2 == 0:
            cell.set_facecolor(COLORS["row_alt"])
        else:
            cell.set_facecolor("white")

        cell.get_text().set_color(COLORS["text"])
        cell.get_text().set_fontsize(10)
        if col == 0:
            cell.get_text().set_ha("left")
        else:
            cell.get_text().set_ha("right")

    for row in range(1, n_rows + 1):
        for col in range(n_cols):
            table[(row, col)].set_height(0.12)


def generate_main_results_table():
    columns = ["Method", "Multi-GT", "Single (Exact)", "Avg Latency"]
    methods = ["se_rag", "bm25_rag", "dense_rag", "tfidf_rag", "no_rag"]
    single_acc = [63.3, 54.9, 30.4, 45.3, 23.5]
    multi_acc = [90.4, 86.9, 76.6, 80.1, 69.8]
    latency_ms = [7090, 8800, 8810, 8860, 7320]
    rows = list(zip(methods, single_acc, multi_acc, latency_ms))
    data = [
        [
            DISPLAY_NAMES.get(method, method),
            f"{multi:.1f}%",
            f"{single:.1f}%",
            f"{latency / 1000:.2f}s",
        ]
        for method, single, multi, latency in rows
    ]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")
    ax.set_title("Table 1: End-to-End Performance Comparison (1000 Queries)", fontsize=12, fontweight="bold", pad=14)

    table = ax.table(
        cellText=data,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.25, 0.22, 0.27, 0.22],
        bbox=[0.03, 0.08, 0.94, 0.80],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    style_table(table, n_rows=len(data), n_cols=len(columns), se_rag_row=1)

    for col in range(len(columns)):
        table[(1, col)].get_text().set_weight("bold")
    table[(1, 1)].get_text().set_weight("bold")

    save_figure(fig, "fig_main_results_table")


def generate_per_action_table():
    columns = ["Action", "N", "No-RAG", "SE-RAG", "Delta"]
    data = [
        ["delay_tolerant", "482", "62.6%", "97.4%", "+34.8pt"],
        ["reassign_order", "228", "78.7%", "85.6%", "+6.9pt"],
        ["ignore", "28", "65.2%", "78.3%", "+13.0pt"],
        ["adjust_capacity", "152", "68.6%", "78.0%", "+9.4pt"],
        ["reroute", "110", "84.8%", "90.2%", "+5.4pt"],
    ]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")
    ax.set_title("Table 2: Per-Action Comparison (1000 Queries)", fontsize=12, fontweight="bold", pad=14)

    table = ax.table(
        cellText=data,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.27, 0.08, 0.17, 0.17, 0.17],
        bbox=[0.03, 0.08, 0.94, 0.80],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    style_table(table, n_rows=len(data), n_cols=len(columns), se_rag_row=None)

    for row_idx, row in enumerate(data, start=1):
        no_rag = float(row[2].rstrip("%"))
        se_rag = float(row[3].rstrip("%"))
        best_col = 3 if se_rag >= no_rag else 2
        table[(row_idx, best_col)].get_text().set_weight("bold")

        delta_cell = table[(row_idx, 4)]
        delta_cell.get_text().set_weight("bold")
        delta_cell.get_text().set_color(COLORS["green"] if row[4].startswith("+") else COLORS["red"])

    save_figure(fig, "fig_per_action_table")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    generators = [
        generate_multi_gt_comparison,
        generate_gt_inconsistency,
        generate_main_results_table,
        generate_per_action_table,
    ]
    for generator in generators:
        generator()

    print("All paper figures and tables generated successfully.")


if __name__ == "__main__":
    main()
