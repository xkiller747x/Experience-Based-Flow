"""SE-RAG Phase 2: Causal Relation Extractor.

Discovers causal relationships between context features and decision
outcomes using conditional mutual information and statistical tests.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from src.rag import Case


# ---------------------------------------------------------------------------
# Discretisation (shared with rule_miner, duplicated for independence)
# ---------------------------------------------------------------------------

def _disc(value: float, n_bins: int = 3, low: float = 0.0, high: float = 1.0) -> int:
    """Simple equal-width binning into n_bins."""
    width = (high - low) / n_bins
    return min(n_bins - 1, max(0, int((value - low) / width)))


FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "severity": (1.0, 8.0),
    "current_load_rate": (0.0, 1.0),
    "available_backup_vehicles": (0.0, 10.0),
    "time_window_pressure": (0.0, 1.0),
    "avg_delay_minutes": (0.0, 120.0),
    "customer_priority_mix": (0.0, 1.0),
    "urgent_orders": (0.0, 30.0),
    "affected_routes": (0.0, 30.0),
}

CAUSAL_FEATURES = [
    "severity",
    "current_load_rate",
    "available_backup_vehicles",
    "time_window_pressure",
    "avg_delay_minutes",
    "customer_priority_mix",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CausalEdge:
    """A directed causal relationship."""

    source: str              # feature name
    target: str              # feature name or "action"
    strength: float          # mutual information or conditional MI
    direction: str = ""      # "positive" / "negative" / ""
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "strength": round(self.strength, 4),
            "direction": self.direction,
            "description": self.description,
        }


@dataclass
class CausalGraph:
    """Directed causal graph over features and action."""

    nodes: list[str] = field(default_factory=list)
    edges: list[CausalEdge] = field(default_factory=list)
    feature_to_action: list[CausalEdge] = field(default_factory=list)
    feature_to_feature: list[CausalEdge] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "edges": [e.to_dict() for e in self.edges],
            "feature_to_action": [e.to_dict() for e in self.feature_to_action],
            "feature_to_feature": [e.to_dict() for e in self.feature_to_feature],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> CausalGraph:
        with Path(path).open(encoding="utf-8") as f:
            data = json.load(f)
        g = cls(nodes=data.get("nodes", []))
        for ed in data.get("edges", []):
            edge = CausalEdge(
                source=ed["source"],
                target=ed["target"],
                strength=ed["strength"],
                direction=ed.get("direction", ""),
                description=ed.get("description", ""),
            )
            g.edges.append(edge)
        for ed in data.get("feature_to_action", []):
            g.feature_to_action.append(CausalEdge(**ed))
        for ed in data.get("feature_to_feature", []):
            g.feature_to_feature.append(CausalEdge(**ed))
        return g

    def top_edges(self, n: int = 20) -> list[CausalEdge]:
        """Return top-n strongest edges."""
        return sorted(self.edges, key=lambda e: -e.strength)[:n]

    def edges_for_feature(self, feature: str) -> list[CausalEdge]:
        """Return edges where feature is source."""
        return [e for e in self.edges if e.source == feature]


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class CausalRelationExtractor:
    """Discover causal relationships between context features and actions.

    Uses:
    1. Mutual information I(feature; action) to measure feature → action
    2. Conditional mutual information I(feature_A; action | feature_B) to
       discover feature → feature → action chains
    """

    def __init__(
        self,
        cases_path: str | Path,
        *,
        n_bins: int = 3,
        mi_threshold: float = 0.01,
        top_k_chains: int = 10,
    ):
        self.n_bins = n_bins
        self.mi_threshold = mi_threshold
        self.top_k_chains = top_k_chains
        self.cases: list[Case] = []
        self._load(cases_path)

    def _load(self, path: str | Path) -> None:
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.cases.append(Case(**json.loads(line)))

    # ---- public API ----

    def extract(self) -> CausalGraph:
        """Build a causal graph from loaded cases."""
        graph = CausalGraph()
        graph.nodes = list(CAUSAL_FEATURES) + ["action"]

        # Step 1: feature → action mutual information
        self._compute_feature_action(graph)

        # Step 2: feature → feature correlations
        self._compute_feature_feature(graph)

        # Step 3: conditional chains feature_A → feature_B → action
        self._compute_conditional_chains(graph)

        return graph

    # ---- internal ----

    def _discretise(self, cases: list[Case]) -> list[dict[str, int]]:
        """Discretise all cases into integer bins."""
        rows = []
        for case in cases:
            row: dict[str, int] = {}
            for feat in CAUSAL_FEATURES:
                val = float(getattr(case, feat, 0))
                low, high = FEATURE_RANGES.get(feat, (0.0, 1.0))
                row[feat] = _disc(val, self.n_bins, low, high)
            # action as int
            action_map = {
                "reroute": 0, "ignore": 1, "adjust_capacity": 2,
                "reassign_order": 3, "delay_tolerant": 4,
            }
            row["action"] = action_map.get(case.action, -1)
            rows.append(row)
        return rows

    def _entropy(self, counter: Counter, total: int) -> float:
        ent = 0.0
        for count in counter.values():
            if count > 0:
                p = count / total
                ent -= p * math.log2(p)
        return ent

    def _mutual_info(self, rows: list[dict], x: str, y: str) -> float:
        """Compute mutual information I(X; Y)."""
        n = len(rows)
        if n == 0:
            return 0.0

        x_counter = Counter(r[x] for r in rows)
        y_counter = Counter(r[y] for r in rows)
        xy_counter = Counter((r[x], r[y]) for r in rows)

        h_x = self._entropy(x_counter, n)
        h_y = self._entropy(y_counter, n)
        h_xy = self._entropy(xy_counter, n)

        return h_x + h_y - h_xy

    def _conditional_mi(self, rows: list[dict], x: str, y: str, z: str) -> float:
        """Compute conditional mutual information I(X; Y | Z)."""
        n = len(rows)
        if n == 0:
            return 0.0

        # Group by z
        by_z: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            by_z[r[z]].append(r)

        cmi = 0.0
        for z_val, group in by_z.items():
            weight = len(group) / n
            mi = self._mutual_info(group, x, y)
            cmi += weight * mi
        return cmi

    def _compute_feature_action(self, graph: CausalGraph) -> None:
        """Step 1: Compute I(feature; action) for each feature."""
        rows = self._discretise(self.cases)
        for feat in CAUSAL_FEATURES:
            mi = self._mutual_info(rows, feat, "action")
            if mi >= self.mi_threshold:
                # Determine direction: correlation between high feature values and specific actions
                direction = self._detect_direction(rows, feat)
                edge = CausalEdge(
                    source=feat,
                    target="action",
                    strength=mi,
                    direction=direction,
                    description=f"I({feat}; action) = {mi:.4f}",
                )
                graph.edges.append(edge)
                graph.feature_to_action.append(edge)

    def _compute_feature_feature(self, graph: CausalGraph) -> None:
        """Step 2: Compute I(feature_A; feature_B) for feature pairs."""
        rows = self._discretise(self.cases)
        for fa, fb in combinations(CAUSAL_FEATURES, 2):
            mi = self._mutual_info(rows, fa, fb)
            if mi >= self.mi_threshold:
                edge = CausalEdge(
                    source=fa,
                    target=fb,
                    strength=mi,
                    description=f"I({fa}; {fb}) = {mi:.4f}",
                )
                graph.edges.append(edge)
                graph.feature_to_feature.append(edge)

                # Also add reverse
                edge_rev = CausalEdge(
                    source=fb,
                    target=fa,
                    strength=mi,
                    description=f"I({fb}; {fa}) = {mi:.4f}",
                )
                graph.edges.append(edge_rev)
                graph.feature_to_feature.append(edge_rev)

    def _compute_conditional_chains(self, graph: CausalGraph) -> None:
        """Step 3: Find mediated chains feature_A → feature_B → action."""
        rows = self._discretise(self.cases)
        chains = []

        for fa in CAUSAL_FEATURES:
            for fb in CAUSAL_FEATURES:
                if fa == fb:
                    continue
                # I(fa; action | fb) — does fa provide info about action beyond fb?
                cmi = self._conditional_mi(rows, fa, "action", fb)
                if cmi >= self.mi_threshold:
                    chains.append((fa, fb, cmi))

        # Sort by strength, add to graph descriptions
        chains.sort(key=lambda t: -t[2])
        for fa, fb, cmi in chains[:self.top_k_chains]:
            # Check if this is a mediation: fa → fb → action
            # If I(fa; action | fb) << I(fa; action), fb mediates
            mi_fa_action = self._mutual_info(rows, fa, "action")
            if mi_fa_action > 0 and cmi / mi_fa_action < 0.5:
                graph.edges.append(CausalEdge(
                    source=fa,
                    target=fb,
                    strength=mi_fa_action - cmi,
                    description=f"Mediated: {fa} → {fb} → action (mediation_ratio={1 - cmi/mi_fa_action:.2f})",
                ))

    def _detect_direction(self, rows: list[dict], feat: str) -> str:
        """Detect if high feature values correlate with specific actions."""
        high_rows = [r for r in rows if r[feat] >= self.n_bins - 1]
        low_rows = [r for r in rows if r[feat] == 0]

        if not high_rows or not low_rows:
            return ""

        high_action = Counter(r["action"] for r in high_rows).most_common(1)[0][0]
        low_action = Counter(r["action"] for r in low_rows).most_common(1)[0][0]

        action_names = {
            0: "reroute", 1: "ignore", 2: "adjust_capacity",
            3: "reassign_order", 4: "delay_tolerant",
        }

        if high_action != low_action:
            return f"{action_names.get(high_action, '?')} when high, {action_names.get(low_action, '?')} when low"
        return ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    cases_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\Code\logistic-ai\data\cases\cases_30k.jsonl"
    output_path = sys.argv[2] if len(sys.argv) > 2 else r"D:\Code\logistic-ai\output\se_rag\causal_graph.json"

    print(f"Loading cases from {cases_path}")
    extractor = CausalRelationExtractor(cases_path)
    print(f"Loaded {len(extractor.cases)} cases")

    print("Extracting causal relations...")
    graph = extractor.extract()
    print(f"Found {len(graph.edges)} edges")
    print(f"  feature → action: {len(graph.feature_to_action)}")
    print(f"  feature ↔ feature: {len(graph.feature_to_feature)}")

    print("\nTop feature → action edges:")
    for edge in sorted(graph.feature_to_action, key=lambda e: -e.strength):
        print(f"  {edge.source} → action: MI={edge.strength:.4f} ({edge.direction})")

    graph.save(output_path)
    print(f"\nSaved causal graph to {output_path}")
