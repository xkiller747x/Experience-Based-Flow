"""
SE-RAG architecture diagram for publication.

Generates:
  - paper/fig_se_rag_architecture.pdf
  - paper/fig_se_rag_architecture.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper"
PNG_PATH = OUT_DIR / "fig_se_rag_architecture.png"
PDF_PATH = OUT_DIR / "fig_se_rag_architecture.pdf"


COLORS = {
    "offline_bg": "#E8F5E9",
    "offline_border": "#2E7D32",
    "online_bg": "#E3F2FD",
    "online_border": "#1565C0",
    "box_fill": "#FFFFFF",
    "text": "#1F2933",
    "muted": "#5F6B73",
    "arrow": "#4B5563",
    "matched": "#1565C0",
    "unmatched": "#6A1B9A",
    "event": "#B85C00",
    "decision": "#B7791F",
}


def add_phase(ax, xy, width, height, label, facecolor, edgecolor):
    """Draw a phase background with a centered title."""
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.035,rounding_size=0.075",
        linewidth=1.8,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=0.72,
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height - 0.22,
        label,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=edgecolor,
        zorder=3,
    )


def add_box(
    ax,
    x,
    y,
    width,
    height,
    title,
    subtitle=None,
    edgecolor="#333333",
    facecolor="#FFFFFF",
    title_size=9,
    subtitle_size=7,
    linewidth=1.8,
):
    """Draw a rounded white box with optional sub-label."""
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.045,rounding_size=0.07",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(patch)

    if subtitle:
        ax.text(
            x + width / 2,
            y + height * 0.60,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=COLORS["text"],
            zorder=4,
        )
        ax.text(
            x + width / 2,
            y + height * 0.32,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=COLORS["muted"],
            zorder=4,
        )
    else:
        ax.text(
            x + width / 2,
            y + height / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=COLORS["text"],
            zorder=4,
        )
    return patch


def arrow(ax, start, end, color=None, lw=1.45, linestyle="-", arrowstyle="-|>", zorder=3):
    """Draw an arrow segment using FancyArrowPatch."""
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=arrowstyle,
        mutation_scale=10.5,
        linewidth=lw,
        linestyle=linestyle,
        color=color or COLORS["arrow"],
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(patch)
    return patch


def route(ax, points, color=None, lw=1.45, linestyle="-", zorder=3):
    """Draw an orthogonal path with the arrowhead only on the final segment."""
    for i, (start, end) in enumerate(zip(points[:-1], points[1:])):
        arrowstyle = "-|>" if i == len(points) - 2 else "-"
        arrow(ax, start, end, color=color, lw=lw, linestyle=linestyle, arrowstyle=arrowstyle, zorder=zorder)


def label(ax, x, y, text, color, size=7, ha="center", va="center"):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=size,
        color=color,
        fontstyle="italic",
        zorder=5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.2),
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.97)

    add_phase(ax, (0.25, 0.42), 4.45, 5.25, "Offline Phase", COLORS["offline_bg"], COLORS["offline_border"])
    add_phase(ax, (5.05, 0.42), 4.70, 5.25, "Online Phase", COLORS["online_bg"], COLORS["online_border"])

    # Offline phase
    add_box(
        ax,
        1.24,
        4.50,
        2.45,
        0.58,
        "Historical Cases",
        "(30k subset)",
        edgecolor=COLORS["offline_border"],
    )
    add_box(
        ax,
        0.62,
        3.62,
        1.72,
        0.58,
        "Rule Mining",
        "rule_miner.py",
        edgecolor=COLORS["offline_border"],
    )
    add_box(
        ax,
        2.62,
        3.62,
        1.72,
        0.58,
        "Causal Extraction",
        "causal_extractor.py",
        edgecolor=COLORS["offline_border"],
    )
    add_box(
        ax,
        0.62,
        2.72,
        1.72,
        0.58,
        "Rule Base",
        "(44k rules)",
        edgecolor=COLORS["offline_border"],
        linewidth=2.0,
    )
    add_box(
        ax,
        2.62,
        2.72,
        1.72,
        0.58,
        "Causal Graph",
        None,
        edgecolor=COLORS["offline_border"],
        linewidth=2.0,
    )

    split = (2.47, 4.32)
    arrow(ax, (2.47, 4.50), split, COLORS["offline_border"], arrowstyle="-")
    route(ax, [split, (1.48, 4.32), (1.48, 4.20)], COLORS["offline_border"])
    route(ax, [split, (3.48, 4.32), (3.48, 4.20)], COLORS["offline_border"])
    route(ax, [(1.48, 3.62), (1.48, 3.30)], COLORS["offline_border"])
    route(ax, [(3.48, 3.62), (3.48, 3.30)], COLORS["offline_border"])

    # Online phase
    add_box(
        ax,
        6.30,
        4.50,
        2.36,
        0.56,
        "New Disruption Event",
        None,
        edgecolor=COLORS["event"],
    )
    add_box(
        ax,
        6.30,
        3.72,
        2.36,
        0.58,
        "Feature Extraction",
        "(24-dim)",
        edgecolor=COLORS["event"],
    )
    add_box(
        ax,
        6.30,
        2.90,
        2.36,
        0.58,
        "Rule Matching",
        "IF-THEN matching",
        edgecolor=COLORS["online_border"],
        linewidth=2.0,
    )
    add_box(
        ax,
        5.36,
        1.47,
        2.72,
        0.76,
        "LLM + Rules + Causal Graph",
        "causal reasoning",
        edgecolor=COLORS["online_border"],
        linewidth=2.0,
    )
    add_box(
        ax,
        8.42,
        1.48,
        1.18,
        0.74,
        "Case Retrieval",
        "fallback\nBM25/BGE",
        edgecolor=COLORS["unmatched"],
        title_size=8.5,
        subtitle_size=6.4,
        linewidth=2.0,
    )
    add_box(
        ax,
        6.26,
        0.63,
        2.52,
        0.60,
        "Dispatch Decision",
        "action",
        edgecolor=COLORS["decision"],
        linewidth=2.2,
    )

    route(ax, [(7.48, 4.50), (7.48, 4.30)], COLORS["arrow"])
    route(ax, [(7.48, 3.72), (7.48, 3.48)], COLORS["arrow"])

    arrow(ax, (7.22, 2.90), (6.72, 2.23), COLORS["matched"], lw=1.8)
    label(ax, 6.92, 2.58, "Matched", COLORS["matched"])

    arrow(ax, (8.23, 2.90), (9.13, 2.22), COLORS["unmatched"], lw=1.8)
    label(ax, 9.10, 2.58, "Unmatched", COLORS["unmatched"], ha="left")

    route(ax, [(8.42, 1.85), (8.08, 1.85)], COLORS["unmatched"], lw=1.55)
    route(ax, [(6.72, 1.47), (6.72, 1.23)], COLORS["arrow"], lw=1.9)

    # Offline artifacts loaded by the online LLM. 1-turn paths:
    # straight down -> horizontal right, entering LLM box from left side
    route(
        ax,
        [(1.48, 2.72), (1.48, 1.85), (5.44, 1.85)],
        COLORS["offline_border"],
        lw=1.35,
        linestyle="--",
        zorder=3,
    )
    route(
        ax,
        [(3.48, 2.72), (3.48, 1.60), (5.44, 1.60)],
        COLORS["offline_border"],
        lw=1.35,
        linestyle="--",
        zorder=3,
    )
    label(ax, 4.55, 1.72, "loaded at startup", COLORS["offline_border"], size=7)

    ax.text(
        5.0,
        5.90,
        "Structured Experience Retrieval-Augmented Generation (SE-RAG)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=COLORS["text"],
        zorder=6,
    )

    fig.savefig(PNG_PATH, dpi=300, facecolor="white")
    fig.savefig(PDF_PATH, dpi=300, facecolor="white")
    plt.close(fig)
    print("DIAGRAM GENERATED SUCCESSFULLY")


if __name__ == "__main__":
    main()
