"""
Visualisation utilities for scheduling experiment results.
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for any environment)
import matplotlib.pyplot as plt
import numpy as np

# Try to load Chinese fonts; fall back to default
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

_COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
_MARKERS = ["o", "s", "^", "D", "v", "P"]


# ---------------------------------------------------------------------------
# 1. Pareto front comparison
# ---------------------------------------------------------------------------

def plot_pareto_fronts(
    results: Dict[str, np.ndarray],
    title: str = "Pareto Front Comparison",
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(10, 7))
    for idx, (name, obj) in enumerate(results.items()):
        c, m = _COLORS[idx % 6], _MARKERS[idx % 6]
        ax.scatter(obj[:, 0], obj[:, 1], label=name, color=c, marker=m, s=60, alpha=0.85, zorder=3)
        si = np.argsort(obj[:, 0])
        ax.plot(obj[si, 0], obj[si, 1], color=c, alpha=0.35, linewidth=1.5)
    ax.set_xlabel("Average Delay (s)", fontsize=13)
    ax.set_ylabel("Total Cost (yuan)", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Convergence curves  (hypervolume)
# ---------------------------------------------------------------------------

def plot_convergence(
    hv_histories: Dict[str, List[float]],
    title: str = "Convergence Comparison (Hypervolume)",
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (name, hv) in enumerate(hv_histories.items()):
        if not hv:
            continue
        ax.plot(hv, label=name, color=_COLORS[idx % 6], linewidth=2)
    ax.set_xlabel("Generation / Iteration", fontsize=13)
    ax.set_ylabel("Hypervolume Indicator", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Bar-chart comparison (delay / cost / SLA)
# ---------------------------------------------------------------------------

def plot_bar_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    metric_labels: Optional[Dict[str, str]] = None,
    title: str = "Algorithm Performance Comparison",
    save_path: Optional[str] = None,
):
    if metric_labels is None:
        metric_labels = {}

    n_m = len(metrics)
    fig, axes = plt.subplots(1, n_m, figsize=(5 * n_m, 6))
    if n_m == 1:
        axes = [axes]

    algo_names = list(results.keys())
    for m_idx, metric in enumerate(metrics):
        ax = axes[m_idx]
        vals  = [results[a].get(metric, 0.0) for a in algo_names]
        bars  = ax.bar(range(len(algo_names)), vals,
                       color=[_COLORS[i % 6] for i in range(len(algo_names))],
                       alpha=0.8, edgecolor="black")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(algo_names)))
        ax.set_xticklabels(algo_names, rotation=30, ha="right", fontsize=9)
        ax.set_title(metric_labels.get(metric, metric), fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Platform load utilisation
# ---------------------------------------------------------------------------

def plot_load_balance(
    chromosomes: Dict[str, np.ndarray],
    tasks,
    platforms,
    title: str = "Platform Load Utilisation",
    save_path: Optional[str] = None,
):
    n_algo = len(chromosomes)
    fig, axes = plt.subplots(1, n_algo, figsize=(6 * n_algo, 5))
    if n_algo == 1:
        axes = [axes]

    caps  = np.array([p.max_capacity for p in platforms])
    names = [p.name for p in platforms]

    for idx, (algo, chrom) in enumerate(chromosomes.items()):
        ax = axes[idx]
        loads = np.bincount(chrom.astype(int),
                            weights=[t.compute_req for t in tasks],
                            minlength=len(platforms))
        util = loads / caps * 100.0
        colors = ["#2ca02c" if u <= 70 else "#ff7f0e" if u <= 90 else "#d62728"
                  for u in util]
        ax.bar(range(len(names)), util, color=colors, alpha=0.85, edgecolor="black")
        ax.axhline(y=100, color="red",   linestyle="--", linewidth=1.5, label="Capacity limit")
        ax.axhline(y=70,  color="green", linestyle=":",  linewidth=1.2, label="Healthy threshold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Utilisation (%)")
        ax.set_ylim(0, 120)
        ax.set_title(algo)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Task-type vs node-type heatmap
# ---------------------------------------------------------------------------

def plot_task_node_heatmap(
    chromosomes: Dict[str, np.ndarray],
    tasks,
    platforms,
    title: str = "Task-Type × Node-Type Assignment Heatmap",
    save_path: Optional[str] = None,
):
    from models import NodeType, TaskType
    task_types   = sorted(set(t.task_type for t in tasks), key=lambda x: x.value)
    node_types   = list(NodeType)
    tt_names     = [tt.name for tt in task_types]
    nt_names     = [nt.name for nt in node_types]

    n_algo = len(chromosomes)
    fig, axes = plt.subplots(1, n_algo, figsize=(6 * n_algo, 5))
    if n_algo == 1:
        axes = [axes]

    for idx, (algo, chrom) in enumerate(chromosomes.items()):
        ax = axes[idx]
        mat = np.zeros((len(task_types), len(node_types)))
        for i, j in enumerate(chrom.astype(int)):
            tt_idx = task_types.index(tasks[i].task_type)
            nt_idx = platforms[j].node_type.value
            mat[tt_idx, nt_idx] += 1

        im = ax.imshow(mat, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(nt_names)));  ax.set_xticklabels(nt_names)
        ax.set_yticks(range(len(tt_names)));  ax.set_yticklabels(tt_names, fontsize=8)
        ax.set_xlabel("Node Type");           ax.set_ylabel("Task Type")
        ax.set_title(algo)
        for r in range(len(task_types)):
            for c in range(len(node_types)):
                v = int(mat[r, c])
                color = "white" if v > mat.max() * 0.6 else "black"
                ax.text(c, r, str(v), ha="center", va="center", fontsize=10, color=color)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Scalability line chart
# ---------------------------------------------------------------------------

def plot_scalability(
    task_sizes: List[int],
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    save_path: Optional[str] = None,
):
    metrics     = ["delay", "cost"]
    metric_lbls = ["Average Delay (s)", "Total Cost (yuan)"]
    fig, axes   = plt.subplots(1, 2, figsize=(14, 6))
    algo_names  = list(next(iter(all_results.values())).keys())

    for m_idx, (metric, label) in enumerate(zip(metrics, metric_lbls)):
        ax = axes[m_idx]
        for a_idx, algo in enumerate(algo_names):
            vals = [all_results[f"N={n}"][algo].get(metric, 0) for n in task_sizes]
            ax.plot(task_sizes, vals, marker="o", label=algo,
                    color=_COLORS[a_idx % 6], linewidth=2)
        ax.set_xlabel("Number of Tasks", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f"Scalability – {metric}", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Scalability Analysis", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
