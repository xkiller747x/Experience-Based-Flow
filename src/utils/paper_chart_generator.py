"""Generate paper-ready charts and tables from raw experiment data.

The script reads ``paper_output/data`` by default and exports figures/tables to
``paper_output/charts`` without changing any project source or runtime state.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402


DEFAULT_DATA_DIR = REPO_ROOT / "paper_output" / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper_output" / "charts"
METHOD_ORDER = (
    "baseline",
    "v1",
    "v2",
    "no_rag",
    "business_rules",
    "ortools_full_replan",
    "bm25_rag",
    "hybrid_rag",
    "rule_prompt",
)
METHOD_LABELS = {
    "baseline": "Baseline",
    "v1": "V1",
    "v2": "V2",
    "no_rag": "No-RAG",
    "business_rules": "Rules",
    "ortools_full_replan": "OR-Tools",
    "bm25_rag": "BM25 RAG",
    "hybrid_rag": "Hybrid RAG",
    "rule_prompt": "Rule Prompt",
}
METHOD_COLORS = {
    "baseline": "#4C78A8",
    "v1": "#F58518",
    "v2": "#54A24B",
    "no_rag": "#B279A2",
    "business_rules": "#E45756",
    "ortools_full_replan": "#72B7B2",
    "bm25_rag": "#FF9DA6",
    "hybrid_rag": "#9D755D",
    "rule_prompt": "#BAB0AC",
}
FALLBACK_COLORS = (
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
)
SCENARIO_ORDER = ("small", "medium", "large", "stress")


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 10,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.6,
            "patch.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_manifest(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "manifest.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def to_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def ordered_methods(rows: list[dict[str, Any]]) -> list[str]:
    available = {row.get("method", "") for row in rows}
    known = [method for method in METHOD_ORDER if method in available]
    unknown = sorted(method for method in available if method and method not in METHOD_ORDER)
    return known + unknown


def method_label(method: str, rows: list[dict[str, Any]] | None = None) -> str:
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    if rows:
        for row in rows:
            if row.get("method") == method and row.get("method_label"):
                return str(row["method_label"])
    return method.replace("_", " ").title()


def method_color(method: str, index: int) -> str:
    return METHOD_COLORS.get(method, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def tidy_method_axis(axis: plt.Axes, method_count: int) -> None:
    if method_count > 4:
        axis.tick_params(axis="x", labelrotation=28)
        for label in axis.get_xticklabels():
            label.set_ha("right")


def method_aware_figsize(figsize: tuple[float, float], method_count: int, scale: float = 0.32) -> tuple[float, float]:
    return max(figsize[0], figsize[0] + max(0, method_count - 4) * scale), figsize[1]


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str]) -> None:
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", format=fmt, dpi=300)
    plt.close(fig)


def annotate_bars(axis: plt.Axes, values: list[float], fmt: str = "{:.1f}") -> None:
    if not axis.patches:
        return
    offset = max(values) * 0.015 if values else 0.5
    for bar, value in zip(axis.patches, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=6.5,
        )


def plot_action_accuracy(rows: list[dict[str, Any]], output_dir: Path, formats: list[str], figsize: tuple[float, float]) -> None:
    methods = ordered_methods(rows)
    lookup = {row["method"]: row for row in rows}
    values = [to_float(lookup[method], "action_accuracy_pct") for method in methods]
    labels = [method_label(method, rows) for method in methods]
    colors = [method_color(method, index) for index, method in enumerate(methods)]

    fig, axis = plt.subplots(figsize=method_aware_figsize(figsize, len(methods)))
    axis.bar(labels, values, color=colors)
    axis.set_ylabel("Action Accuracy (%)")
    axis.set_xlabel("Retrieval Method")
    axis.set_ylim(0, max(100.0, max(values, default=0.0) * 1.18))
    axis.set_title("Action Accuracy Comparison")
    tidy_method_axis(axis, len(methods))
    annotate_bars(axis, values)
    save_figure(fig, output_dir, "fig_action_accuracy", formats)


def plot_ndcg(rows: list[dict[str, Any]], output_dir: Path, formats: list[str], figsize: tuple[float, float]) -> None:
    methods = ordered_methods(rows)
    lookup = {row["method"]: row for row in rows}
    values = [to_float(lookup[method], "ndcg_at_5_pct") for method in methods]
    labels = [method_label(method, rows) for method in methods]
    colors = [method_color(method, index) for index, method in enumerate(methods)]

    fig, axis = plt.subplots(figsize=method_aware_figsize(figsize, len(methods)))
    axis.bar(labels, values, color=colors)
    axis.set_ylabel("NDCG@5 (%)")
    axis.set_xlabel("Retrieval Method")
    axis.set_ylim(0, max(100.0, max(values, default=0.0) * 1.18))
    axis.set_title("NDCG@5 Comparison")
    tidy_method_axis(axis, len(methods))
    annotate_bars(axis, values)
    save_figure(fig, output_dir, "fig_ndcg_at_5", formats)


def plot_latency(rows: list[dict[str, Any]], output_dir: Path, formats: list[str], figsize: tuple[float, float]) -> None:
    methods = ordered_methods(rows)
    lookup = {row["method"]: row for row in rows}
    values = [max(to_float(lookup[method], "avg_latency_ms"), 1e-6) for method in methods]
    labels = [method_label(method, rows) for method in methods]
    colors = [method_color(method, index) for index, method in enumerate(methods)]

    fig, axis = plt.subplots(figsize=method_aware_figsize(figsize, len(methods)))
    axis.bar(labels, values, color=colors)
    axis.set_ylabel("Decision Latency (ms, log scale)")
    axis.set_xlabel("Retrieval Method")
    axis.set_yscale("log")
    axis.set_title("Decision Latency Comparison")
    tidy_method_axis(axis, len(methods))
    annotate_bars(axis, values, "{:.4f}")
    save_figure(fig, output_dir, "fig_decision_latency", formats)


def plot_scenario_performance(rows: list[dict[str, Any]], output_dir: Path, formats: list[str], figsize: tuple[float, float]) -> None:
    scenarios = [scenario for scenario in SCENARIO_ORDER if any(row["scenario"] == scenario for row in rows)]
    methods = ordered_methods(rows)
    lookup = {(row["scenario"], row["method"]): row for row in rows}

    fig, axis = plt.subplots(figsize=method_aware_figsize(figsize, len(methods), scale=0.22))
    x = np.arange(len(scenarios))
    for index, method in enumerate(methods):
        values = [to_float(lookup.get((scenario, method), {}), "action_accuracy_pct") for scenario in scenarios]
        axis.plot(x, values, marker="o", label=method_label(method, rows), color=method_color(method, index))
    axis.set_xticks(x)
    axis.set_xticklabels([scenario.title() for scenario in scenarios])
    axis.set_ylabel("Action Accuracy (%)")
    axis.set_xlabel("Scenario Scale")
    axis.set_ylim(0, 105)
    axis.set_title("Scenario Performance Curve")
    axis.legend(frameon=True, ncol=2 if len(methods) > 4 else 1)
    save_figure(fig, output_dir, "fig_scenario_performance", formats)


def plot_anomaly_distribution(rows: list[dict[str, Any]], output_dir: Path, formats: list[str], figsize: tuple[float, float]) -> None:
    methods = ordered_methods(rows)
    event_types = sorted({row["event_type"] for row in rows})
    lookup = {(row["event_type"], row["method"]): row for row in rows}
    matrix = np.array(
        [[to_float(lookup.get((event_type, method), {}), "action_accuracy_pct") for method in methods] for event_type in event_types]
    )

    height = max(figsize[1], 0.22 * len(event_types) + 1.6)
    fig, axis = plt.subplots(figsize=(method_aware_figsize(figsize, len(methods), scale=0.22)[0], height))
    image = axis.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    axis.set_xticks(np.arange(len(methods)))
    axis.set_xticklabels([method_label(method, rows) for method in methods])
    axis.set_yticks(np.arange(len(event_types)))
    axis.set_yticklabels([event_type.replace("_", " ") for event_type in event_types])
    axis.set_xlabel("Retrieval Method")
    axis.set_ylabel("Anomaly Type")
    axis.set_title("Anomaly Type Accuracy Distribution")
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("Accuracy (%)")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.0f}", ha="center", va="center", fontsize=5.5)
    save_figure(fig, output_dir, "fig_anomaly_accuracy_distribution", formats)


def plot_retrieval_quality(rows: list[dict[str, Any]], output_dir: Path, formats: list[str], figsize: tuple[float, float]) -> None:
    methods = ordered_methods(rows)
    k_values = sorted({int(row["k"]) for row in rows})
    lookup = {(row["method"], int(row["k"])): row for row in rows}

    quality_width = method_aware_figsize(figsize, len(methods), scale=0.26)[0] * 1.55
    fig, axes = plt.subplots(1, 2, figsize=(quality_width, figsize[1]), constrained_layout=True)
    for index, method in enumerate(methods):
        recall = [to_float(lookup.get((method, k), {}), "recall_at_k_pct") for k in k_values]
        precision = [to_float(lookup.get((method, k), {}), "precision_at_k_pct") for k in k_values]
        axes[0].plot(k_values, recall, marker="o", label=method_label(method, rows), color=method_color(method, index))
        axes[1].plot(k_values, precision, marker="s", label=method_label(method, rows), color=method_color(method, index))

    axes[0].set_title("Recall@k")
    axes[0].set_ylabel("Recall (%)")
    axes[0].set_xlabel("k")
    axes[0].set_ylim(0, 105)
    axes[0].set_xticks(k_values)
    axes[0].legend(frameon=True, ncol=2 if len(methods) > 4 else 1)

    axes[1].set_title("Precision@k")
    axes[1].set_ylabel("Precision (%)")
    axes[1].set_xlabel("k")
    axes[1].set_ylim(0, 105)
    axes[1].set_xticks(k_values)
    axes[1].legend(frameon=True, ncol=2 if len(methods) > 4 else 1)
    save_figure(fig, output_dir, "fig_retrieval_quality", formats)


def plot_system_architecture(output_dir: Path, formats: list[str], figsize: tuple[float, float]) -> None:
    fig, axis = plt.subplots(figsize=(figsize[0] * 1.45, figsize[1] * 0.95))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    def draw_box(
        label: str,
        x_pos: float,
        y_pos: float,
        width: float,
        height: float,
        color: str,
        fontsize: float = 7.0,
    ) -> None:
        box = FancyBboxPatch(
            (x_pos - width / 2, y_pos - height / 2),
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            facecolor=color,
            edgecolor="#333333",
            linewidth=0.8,
            transform=axis.transAxes,
            clip_on=False,
        )
        axis.add_patch(box)
        axis.text(
            x_pos,
            y_pos,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            linespacing=1.15,
            transform=axis.transAxes,
        )

    def draw_arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords="axes fraction",
            arrowprops={
                "arrowstyle": "->",
                "lw": 0.9,
                "color": "#333333",
                "shrinkA": 2,
                "shrinkB": 2,
            },
        )

    boxes = [
        ("Simulation\nvehicles / orders / events", 0.17, 0.72, 0.25, 0.18, "#DCEBFF"),
        ("OR-Tools VRP\ncapacity + time windows", 0.50, 0.72, 0.25, 0.18, "#E8F5E9"),
        ("Route State\ndistance / unassigned", 0.83, 0.72, 0.25, 0.18, "#E0F7FA"),
        ("Case + Rule Memory\nhistorical actions", 0.17, 0.40, 0.25, 0.18, "#FFF3E0"),
        ("Experience Retriever\nBaseline / V1 / V2", 0.50, 0.40, 0.25, 0.18, "#F3E5F5"),
        ("Retrieved Evidence\ntop-k cases + rules", 0.83, 0.40, 0.25, 0.18, "#FCE4EC"),
        ("LLM Agent\nstructured JSON output", 0.50, 0.13, 0.28, 0.17, "#EEF2FF"),
        ("Dispatch Action\nreroute / reassign / delay", 0.83, 0.13, 0.25, 0.17, "#EAF7EA"),
    ]
    for box in boxes:
        draw_box(*box)

    arrows = [
        ((0.30, 0.72), (0.37, 0.72)),
        ((0.63, 0.72), (0.70, 0.72)),
        ((0.30, 0.40), (0.37, 0.40)),
        ((0.63, 0.40), (0.70, 0.40)),
        ((0.50, 0.31), (0.50, 0.22)),
        ((0.64, 0.13), (0.70, 0.13)),
        ((0.83, 0.63), (0.83, 0.49)),
        ((0.83, 0.31), (0.83, 0.22)),
    ]
    for arrow in arrows:
        draw_arrow(*arrow)

    axis.text(
        0.50,
        0.94,
        "Experience-Driven Logistics Disruption Management Architecture",
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        transform=axis.transAxes,
    )
    save_figure(fig, output_dir, "fig_system_architecture", formats)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_latex_table(path: Path, rows: list[dict[str, Any]], caption: str, label: str) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    align = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(latex_escape(column) for column in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(column, "")) for column in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_tables(
    model_rows: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    model_table = [
        {
            "Method": METHOD_LABELS.get(row["method"], row["method"]),
            r"Action Acc. (\%)": row["action_accuracy_pct"],
            r"NDCG@5 (\%)": row["ndcg_at_5_pct"],
            "Latency (ms)": row["avg_latency_ms"],
        }
        for row in model_rows
    ]
    scenario_table = [
        {
            "Scenario": row["scenario"].title(),
            "Method": METHOD_LABELS.get(row["method"], row["method"]),
            r"Action Acc. (\%)": row["action_accuracy_pct"],
            r"NDCG@5 (\%)": row["ndcg_at_5_pct"],
            "Latency (ms)": row["avg_latency_ms"],
        }
        for row in scenario_rows
    ]
    quality_table = [
        {
            "Method": METHOD_LABELS.get(row["method"], row["method"]),
            "k": row["k"],
            r"Recall@k (\%)": row["recall_at_k_pct"],
            r"Precision@k (\%)": row["precision_at_k_pct"],
            r"Hit Rate (\%)": row["hit_rate_at_k_pct"],
        }
        for row in quality_rows
    ]

    write_csv(table_dir / "table_model_comparison.csv", model_table)
    write_csv(table_dir / "table_scenario_performance.csv", scenario_table)
    write_csv(table_dir / "table_retrieval_quality.csv", quality_table)
    write_latex_table(table_dir / "table_model_comparison.tex", model_table, "Retrieval model comparison", "tab:model-comparison")
    write_latex_table(table_dir / "table_scenario_performance.tex", scenario_table, "Scenario performance comparison", "tab:scenario-performance")
    write_latex_table(table_dir / "table_retrieval_quality.tex", quality_table, "Retrieval quality comparison", "tab:retrieval-quality")


def parse_formats(value: str) -> list[str]:
    formats = [item.strip().lower() for item in value.split(",") if item.strip()]
    allowed = {"png", "svg", "pdf"}
    invalid = sorted(set(formats) - allowed)
    if invalid:
        raise ValueError(f"Unsupported formats: {invalid}; allowed: {sorted(allowed)}")
    return formats or ["png", "svg", "pdf"]


def parse_size(value: str) -> tuple[float, float]:
    separator = "x" if "x" in value.lower() else ","
    parts = [part.strip() for part in value.lower().split(separator) if part.strip()]
    if len(parts) != 2:
        raise ValueError("Figure size must look like '3.5x2.6' or '3.5,2.6'.")
    width, height = float(parts[0]), float(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError("Figure dimensions must be positive.")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render paper-ready charts and LaTeX/CSV tables.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing raw experiment CSV/JSON files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for chart/table exports.")
    parser.add_argument("--formats", default="png,svg,pdf", help="Comma-separated figure formats: png,svg,pdf.")
    parser.add_argument("--figsize", default="3.5x2.6", help="Base IEEE-style figure size in inches.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = parse_formats(args.formats)
    figsize = parse_size(args.figsize)

    model_rows = read_csv_rows(data_dir / "retrieval_model_comparison.csv")
    scenario_rows = read_csv_rows(data_dir / "scenario_performance.csv")
    anomaly_rows = read_csv_rows(data_dir / "anomaly_type_accuracy.csv")
    quality_rows = read_csv_rows(data_dir / "retrieval_quality.csv")
    manifest = read_manifest(data_dir)

    plot_action_accuracy(model_rows, output_dir, formats, figsize)
    plot_ndcg(model_rows, output_dir, formats, figsize)
    plot_latency(model_rows, output_dir, formats, figsize)
    plot_scenario_performance(scenario_rows, output_dir, formats, figsize)
    plot_anomaly_distribution(anomaly_rows, output_dir, formats, figsize)
    plot_retrieval_quality(quality_rows, output_dir, formats, figsize)
    plot_system_architecture(output_dir, formats, figsize)
    export_tables(model_rows, scenario_rows, quality_rows, output_dir)

    summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "formats": formats,
        "figsize": figsize,
        "source_manifest": manifest,
        "figures": [
            "fig_action_accuracy",
            "fig_ndcg_at_5",
            "fig_decision_latency",
            "fig_scenario_performance",
            "fig_anomaly_accuracy_distribution",
            "fig_retrieval_quality",
            "fig_system_architecture",
        ],
        "tables_dir": str(output_dir / "tables"),
    }
    with (output_dir / "chart_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"Charts written to: {output_dir}")
    print(f"Tables written to: {output_dir / 'tables'}")


if __name__ == "__main__":
    main()
