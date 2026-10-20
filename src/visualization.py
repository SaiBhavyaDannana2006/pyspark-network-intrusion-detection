"""
==============================================================================
SentinelSpark — High-Resolution Visualization & Reporting Engine
==============================================================================
Generates publication-quality charts using Matplotlib and Plotly across 8
core operational intrusion analytics:
1. normal_vs_attack_dist.png
2. attack_category_dist.png
3. model_comparison_metrics.png
4. binary_confusion_matrix.png
5. multiclass_confusion_matrix.png
6. feature_importance_top20.png
7. per_class_f1_scores.png
8. protocol_attack_rate.png
==============================================================================
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for automated pipelines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import logger, PROJECT_ROOT


class Visualizer:
    """Generates publication-quality security and ML evaluation figures."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        figures_dir: Optional[Union[str, Path]] = None,
    ):
        self.config = config or {}
        if figures_dir is not None:
            self.figures_dir = Path(figures_dir)
        elif self.config.get("models", {}).get("results_dir"):
            self.figures_dir = Path(self.config["models"]["results_dir"])
        elif self.config.get("data", {}).get("figures_dir"):
            self.figures_dir = PROJECT_ROOT / self.config["data"]["figures_dir"]
        else:
            self.figures_dir = PROJECT_ROOT / "results/figures"

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self._setup_style()

    def _setup_style(self) -> None:
        """Apply sleek modern typography and dark cybersecurity aesthetic."""
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.autolayout": True,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.edgecolor": "#d0d7de",
            "axes.linewidth": 0.8,
        })

    def plot_normal_vs_attack_distribution(
        self,
        target_df: pd.DataFrame,
        filename: str = "normal_vs_attack_dist.png",
    ) -> str:
        """Plot binary flow distribution (Normal vs Intrusion)."""
        out_path = self.figures_dir / filename
        fig, ax = plt.subplots(figsize=(6, 5))

        labels = (
            target_df["class_name"].tolist()
            if "class_name" in target_df.columns
            else target_df["label"].astype(str).tolist()
        )
        counts = target_df["count"].tolist()
        total = sum(counts)
        colors = ["#2da44e", "#cf222e"] if len(counts) == 2 else ["#0969da", "#8250df"]

        bars = ax.bar(labels, counts, color=colors, width=0.55, edgecolor="#24292f", linewidth=0.8)

        for bar in bars:
            h = bar.get_height()
            pct = (h / total) * 100 if total > 0 else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (max(counts) * 0.02),
                f"{h:,}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )

        ax.set_title("Network Flow Distribution: Normal vs Malicious", pad=15, fontweight="bold")
        ax.set_ylabel("Total Flow Records")
        ax.set_ylim(0, max(counts) * 1.2)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved binary distribution figure: {out_path}")
        return str(out_path)

    def plot_attack_category_distribution(
        self,
        cat_df: pd.DataFrame,
        filename: str = "attack_category_dist.png",
    ) -> str:
        """Plot multiclass attack category frequency distribution."""
        out_path = self.figures_dir / filename
        fig, ax = plt.subplots(figsize=(8, 5))

        df_sorted = cat_df.sort_values(by="count", ascending=True)
        categories = df_sorted["attack_cat"].tolist()
        counts = df_sorted["count"].tolist()
        total = sum(counts)

        colors = ["#2da44e" if c == "Normal" else "#8250df" for c in categories]
        bars = ax.barh(categories, counts, color=colors, height=0.65, edgecolor="#24292f", linewidth=0.6)

        for bar in bars:
            w = bar.get_width()
            pct = (w / total) * 100 if total > 0 else 0
            ax.text(
                w + (max(counts) * 0.015),
                bar.get_y() + bar.get_height() / 2,
                f" {w:,} ({pct:.1f}%)",
                va="center",
                fontsize=8.5,
                fontweight="bold",
            )

        ax.set_title("UNSW-NB15 Attack Category Profile", pad=15, fontweight="bold")
        ax.set_xlabel("Total Flow Records")
        ax.set_xlim(0, max(counts) * 1.25)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved attack category distribution figure: {out_path}")
        return str(out_path)

    def plot_model_comparison_metrics(
        self,
        metrics_map: Dict[str, Dict[str, float]],
        filename: str = "model_comparison_metrics.png",
    ) -> str:
        """Plot comparative performance metrics across Stage 1 binary classifiers."""
        out_path = self.figures_dir / filename
        fig, ax = plt.subplots(figsize=(9, 5))

        models = list(metrics_map.keys())
        metric_keys = ["attack_recall", "f1_score", "attack_precision", "pr_auc", "roc_auc"]
        display_names = ["Recall", "F1-Score", "Precision", "PR-AUC", "ROC-AUC"]

        x = np.arange(len(display_names))
        width = 0.8 / len(models)
        palette = ["#0969da", "#8250df", "#1f883d", "#bc4c00"]

        for idx, model_name in enumerate(models):
            m = metrics_map[model_name]
            vals = [m.get(k, m.get(k.replace("attack_", ""), 0.0)) for k in metric_keys]
            offset = (idx - (len(models) - 1) / 2) * width
            ax.bar(
                x + offset,
                vals,
                width,
                label=model_name,
                color=palette[idx % len(palette)],
                edgecolor="#24292f",
                linewidth=0.6,
            )

        ax.set_title("Stage 1 Classifier Comparative Performance", pad=15, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, fontweight="bold")
        ax.set_ylabel("Metric Score")
        ax.set_ylim(0, 1.15)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved model comparison figure: {out_path}")
        return str(out_path)

    def plot_binary_confusion_matrix(
        self,
        cm_dict: Dict[str, int],
        filename: str = "binary_confusion_matrix.png",
    ) -> str:
        """Plot 2x2 confusion matrix (TN, FP, FN, TP) with annotated metrics."""
        out_path = self.figures_dir / filename
        fig, ax = plt.subplots(figsize=(5.5, 4.8))

        tn = cm_dict.get("tn", 0)
        fp = cm_dict.get("fp", 0)
        fn = cm_dict.get("fn", 0)
        tp = cm_dict.get("tp", 0)
        matrix = np.array([[tn, fp], [fn, tp]])
        total = max(1, tn + fp + fn + tp)

        cax = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

        labels = [
            [f"True Negative\n{tn:,}\n({tn/total*100:.1f}%)", f"False Positive\n{fp:,}\n({fp/total*100:.1f}%)"],
            [f"False Negative\n{fn:,}\n({fn/total*100:.1f}%)", f"True Positive\n{tp:,}\n({tp/total*100:.1f}%)"],
        ]

        for i in range(2):
            for j in range(2):
                color = "white" if matrix[i, j] > total * 0.4 else "black"
                ax.text(j, i, labels[i][j], ha="center", va="center", color=color, fontweight="bold", fontsize=9.5)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Normal (0)", "Pred Attack (1)"], fontweight="bold")
        ax.set_yticklabels(["Actual Normal (0)", "Actual Attack (1)"], fontweight="bold")
        ax.set_title("Stage 1 Binary Confusion Matrix", pad=15, fontweight="bold")

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved binary confusion matrix: {out_path}")
        return str(out_path)

    def plot_multiclass_confusion_matrix(
        self,
        cm_data: Union[pd.DataFrame, Dict[str, Dict[str, int]]],
        filename: str = "multiclass_confusion_matrix.png",
    ) -> str:
        """Plot multiclass threat taxonomy confusion matrix."""
        out_path = self.figures_dir / filename
        fig, ax = plt.subplots(figsize=(8, 6.5))

        if isinstance(cm_data, dict):
            cm_data = pd.DataFrame(cm_data)

        matrix = cm_data.values
        classes = list(cm_data.index)

        cax = ax.imshow(matrix, cmap="Purples", interpolation="nearest")
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

        for i in range(len(classes)):
            for j in range(len(cm_data.columns)):
                val = int(matrix[i, j])
                color = "white" if val > (matrix.max() * 0.5) else "black"
                if val > 0:
                    ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=8.5, fontweight="bold")

        ax.set_xticks(range(len(cm_data.columns)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(cm_data.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(classes, fontsize=9)
        ax.set_xlabel("Predicted Attack Category", fontweight="bold", labelpad=10)
        ax.set_ylabel("Actual Attack Category", fontweight="bold", labelpad=10)
        ax.set_title("Stage 2 Multiclass Taxonomy Confusion Matrix", pad=15, fontweight="bold")

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved multiclass confusion matrix: {out_path}")
        return str(out_path)

    def plot_feature_importance_top20(
        self,
        imp_df: pd.DataFrame,
        filename: str = "feature_importance_top20.png",
    ) -> str:
        """Plot top 20 most predictive network features."""
        out_path = self.figures_dir / filename
        fig, ax = plt.subplots(figsize=(8, 6))

        top20 = imp_df.head(20).sort_values(by="importance", ascending=True)
        features = top20["feature"].tolist()
        importances = top20["importance"].tolist()

        bars = ax.barh(features, importances, color="#0969da", height=0.65, edgecolor="#24292f", linewidth=0.6)

        max_val = max(importances) if importances else 1.0
        for bar in bars:
            w = bar.get_width()
            ax.text(
                w + (max_val * 0.015),
                bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

        ax.set_title("Top 20 Predictive Network Flow Features (Gini Importance)", pad=15, fontweight="bold")
        ax.set_xlabel("Importance Score")
        ax.set_xlim(0, max_val * 1.2)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved top 20 feature importance figure: {out_path}")
        return str(out_path)

    def plot_per_class_f1_scores(
        self,
        f1_map: Dict[str, Dict[str, float]],
        filename: str = "per_class_f1_scores.png",
    ) -> str:
        """Plot per-category F1 scores for the attack taxonomy."""
        out_path = self.figures_dir / filename
        fig, ax = plt.subplots(figsize=(8, 4.8))

        classes = list(f1_map.keys())
        f1_scores = [
            f1_map[c].get("f1", f1_map[c].get("f1_score", 0.0))
            if isinstance(f1_map[c], dict)
            else float(f1_map[c])
            for c in classes
        ]

        df = pd.DataFrame({"class": classes, "f1": f1_scores}).sort_values(by="f1", ascending=True)
        colors = ["#2da44e" if score >= 0.8 else "#bf8700" if score >= 0.6 else "#cf222e" for score in df["f1"]]

        bars = ax.barh(df["class"], df["f1"], color=colors, height=0.6, edgecolor="#24292f", linewidth=0.6)

        for bar in bars:
            w = bar.get_width()
            ax.text(
                w + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.2f}",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        ax.axvline(0.80, color="#2da44e", linestyle=":", linewidth=1.2, label="High Precision Target (0.80)")
        ax.set_title("Per-Class F1 Score Breakdown Across Attack Taxonomy", pad=15, fontweight="bold")
        ax.set_xlabel("F1-Score")
        ax.set_xlim(0, 1.15)
        ax.legend(loc="lower right")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved per-class F1 score figure: {out_path}")
        return str(out_path)

    def plot_protocol_attack_rate(
        self,
        proto_df: pd.DataFrame,
        filename: str = "protocol_attack_rate.png",
    ) -> str:
        """Plot attack density by transport protocol."""
        out_path = self.figures_dir / filename
        fig, ax = plt.subplots(figsize=(7, 4.8))

        df_sorted = proto_df.sort_values(by="attack_rate", ascending=False)
        protos = df_sorted["proto"].tolist()
        rates = df_sorted["attack_rate"].tolist()

        colors = ["#cf222e" if r >= 50 else "#0969da" for r in rates]
        bars = ax.bar(protos, rates, color=colors, width=0.55, edgecolor="#24292f", linewidth=0.6)

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.5,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )

        ax.set_title("Intrusion Attack Rate by Network Transport Protocol", pad=15, fontweight="bold")
        ax.set_ylabel("Attack Rate (%)")
        ax.set_ylim(0, max(rates, default=100) * 1.25)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved protocol attack rate figure: {out_path}")
        return str(out_path)


# Backward-compatible alias
PipelineVisualizer = Visualizer
