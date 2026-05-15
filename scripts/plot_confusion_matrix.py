"""
Generate the SE-RAG 1000-query confusion matrix figure.

Outputs:
  - paper/fig_confusion_matrix.png
  - paper/fig_confusion_matrix.pdf
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper"

COLORS = {
    "primary": "#2E86AB",
    "accent": "#A23B72",
    "highlight": "#F18F01",
    "warm": "#C9A66B",
    "text": "#2D3436",
    "grid": "#E0E0E0",
}

ACTIONS = ["delay_tolerant", "adjust_capacity", "reassign_order", "reroute", "ignore"]
CM_DATA = np.array(
    [
        [334, 26, 59, 43, 9],
        [56, 77, 7, 2, 4],
        [75, 17, 117, 11, 0],
        [26, 3, 0, 79, 0],
        [12, 0, 4, 0, 11],
    ],
    dtype=int,
)

CLASSIFIED_TOTAL = int(CM_DATA.sum())
PARSING_ERRORS = 28
TOTAL_QUERIES = CLASSIFIED_TOTAL + PARSING_ERRORS
CORRECT = int(np.trace(CM_DATA))
ACCURACY = CORRECT / CLASSIFIED_TOTAL


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


def pretty_action(action):
    return action.replace("_", " ").title()


def annotate_cells(ax, cm):
    row_totals = cm.sum(axis=1)
    threshold = cm.max() * 0.48

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            pct = count / row_totals[i] * 100 if row_totals[i] else 0.0
            text_color = "white" if count >= threshold else COLORS["text"]
            fontweight = "bold" if i == j else "normal"
            ax.text(
                j,
                i,
                f"{count}\n{pct:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight=fontweight,
                linespacing=1.35,
            )


def highlight_diagonal(ax, size):
    for idx in range(size):
        ax.add_patch(
            Rectangle(
                (idx - 0.5, idx - 0.5),
                1,
                1,
                fill=False,
                edgecolor=COLORS["highlight"],
                linewidth=2.4,
                zorder=3,
            )
        )


def plot_confusion_matrix():
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = [pretty_action(action) for action in ACTIONS]
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    image = ax.imshow(CM_DATA, cmap="Blues", interpolation="nearest")

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count", color=COLORS["text"])
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    cbar.ax.tick_params(colors=COLORS["text"])

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="left", rotation_mode="anchor", fontsize=9)
    ax.set_yticklabels(labels, fontsize=10)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Predicted Action", labelpad=14, fontsize=11)
    ax.set_ylabel("True Action", fontsize=11)
    ax.set_title("Confusion Matrix - SE-RAG (1000 Queries)", fontsize=13, fontweight="bold", pad=34)

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="x", bottom=False, top=True, labeltop=True, labelbottom=False)

    annotate_cells(ax, CM_DATA)
    highlight_diagonal(ax, len(labels))

    ax.text(
        0.5,
        -0.12,
        f"{ACCURACY * 100:.1f}% accuracy ({PARSING_ERRORS}/{TOTAL_QUERIES} parsing errors excluded)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color=COLORS["text"],
    )

    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
        spine.set_linewidth(1.0)

    png_path = OUT_DIR / "fig_confusion_matrix.png"
    pdf_path = OUT_DIR / "fig_confusion_matrix.pdf"
    fig.tight_layout(pad=1.2)
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {png_path.relative_to(ROOT)}")
    print(f"Saved {pdf_path.relative_to(ROOT)}")
    print(f"Classified total: {CLASSIFIED_TOTAL}, correct: {CORRECT}, accuracy: {ACCURACY * 100:.1f}%")


if __name__ == "__main__":
    plot_confusion_matrix()
