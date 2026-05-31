"""
Experiment driver: runs all algorithms, collects metrics, saves plots and CSV.
"""

import os
import time
from typing import Dict, List, Tuple

import numpy as np

from baselines import GeneticAlgo, MinMin, MOPSO, PSO, StandardNSGAII
from data_generator import generate_scenario
from dw_topsis import DWTOPSIS
from models import TaskType
from nsga2_improved import ImprovedNSGAII
from problem import SchedulingProblem
from visualization import (
    plot_bar_comparison,
    plot_convergence,
    plot_load_balance,
    plot_pareto_fronts,
    plot_scalability,
    plot_task_node_heatmap,
)

os.makedirs("results", exist_ok=True)


# ============================================================
# Helper
# ============================================================

def _best_from_pareto(pareto_pop, pareto_obj, task_types, problem):
    """Use DW-TOPSIS to pick the best solution from a Pareto set."""
    dw = DWTOPSIS(task_types)
    idx = dw.decide(pareto_pop, pareto_obj)
    chrom = pareto_pop[idx]
    f1, f2, _ = problem.evaluate(chrom)
    return chrom, f1, f2, dw


def _print_table(results: Dict[str, Dict[str, float]], title: str):
    sep = "=" * 70
    print(f"\n{sep}\n{title}\n{sep}")
    print(f"{'Algorithm':<30} {'Avg Delay (s)':<16} {'Total Cost':<14} {'SLA Sat.'}")
    print("-" * 70)
    for algo, m in results.items():
        print(f"{algo:<30} {m['delay']:<16.4f} {m['cost']:<14.4f} {m.get('sla', 0):.2%}")


# ============================================================
# Single experiment
# ============================================================

def run_single_experiment(
    n_tasks: int = 50,
    n_core: int = 3,
    n_regional: int = 4,
    n_edge: int = 5,
    pop_size: int = 100,
    n_gen: int = 200,
    preference: np.ndarray = None,
    alpha: float = 0.5,
    seed: int = 42,
    scenario_name: str = "main",
) -> Dict:
    if preference is None:
        preference = np.array([0.5, 0.5])

    print(f"\n{'='*60}")
    print(f"Scenario : {scenario_name}")
    print(f"Tasks    : {n_tasks}  |  Platforms : {n_core+n_regional+n_edge}")
    print(f"Pop size : {pop_size} |  Generations: {n_gen}")
    print(f"Preference vector: {preference}")
    print(f"{'='*60}")

    tasks, platforms = generate_scenario(n_tasks, n_core, n_regional, n_edge, seed=seed)
    problem = SchedulingProblem(tasks, platforms)
    task_types = [t.task_type for t in tasks]

    results: Dict[str, Dict[str, float]] = {}
    pareto_for_plot: Dict[str, np.ndarray] = {}
    hv_histories: Dict[str, List[float]] = {}
    best_chroms: Dict[str, np.ndarray] = {}

    # ── 1. Improved NSGA-II + DW-TOPSIS ──────────────────────────────────
    print("\n[1/6] Improved NSGA-II + DW-TOPSIS")
    t0 = time.time()
    alg1 = ImprovedNSGAII(problem, pop_size=pop_size, n_generations=n_gen,
                           preference=preference, alpha=alpha, seed=seed)
    pp, po = alg1.run()
    chrom1, f1, f2, dw = _best_from_pareto(pp, po, task_types, problem)
    sla1 = problem.sla_satisfaction_rate(chrom1)
    results["Imp.NSGA-II+DW-TOPSIS"]    = {"delay": f1, "cost": f2, "sla": sla1}
    pareto_for_plot["Imp.NSGA-II+DW-TOPSIS"] = po
    hv_histories["Imp.NSGA-II+DW-TOPSIS"]    = alg1.history["hypervolume"]
    best_chroms["Imp.NSGA-II+DW-TOPSIS"]     = chrom1
    w = dw.weight_report(po)
    print(f"   delay={f1:.4f}s  cost={f2:.4f}  SLA={sla1:.2%}  [{time.time()-t0:.1f}s]")
    print(f"   DW-TOPSIS weights: subj={w['subjective'].round(3)}  "
          f"obj={w['objective'].round(3)}  comb={w['combined'].round(3)}")

    # ── 2. Standard NSGA-II + DW-TOPSIS ──────────────────────────────────
    print("\n[2/6] Standard NSGA-II + DW-TOPSIS")
    t0 = time.time()
    alg2 = StandardNSGAII(problem, pop_size=pop_size, n_gen=n_gen, seed=seed)
    pp2, po2 = alg2.run()
    chrom2, f1b, f2b, _ = _best_from_pareto(pp2, po2, task_types, problem)
    sla2 = problem.sla_satisfaction_rate(chrom2)
    results["Std.NSGA-II+DW-TOPSIS"]    = {"delay": f1b, "cost": f2b, "sla": sla2}
    pareto_for_plot["Std.NSGA-II+DW-TOPSIS"] = po2
    hv_histories["Std.NSGA-II+DW-TOPSIS"]    = alg2.history["hypervolume"]
    best_chroms["Std.NSGA-II+DW-TOPSIS"]     = chrom2
    print(f"   delay={f1b:.4f}s  cost={f2b:.4f}  SLA={sla2:.2%}  [{time.time()-t0:.1f}s]")

    # ── 3. MOPSO + DW-TOPSIS ─────────────────────────────────────────────
    print("\n[3/6] MOPSO + DW-TOPSIS")
    t0 = time.time()
    alg3 = MOPSO(problem, n_particles=pop_size, n_iter=n_gen, seed=seed)
    pp3, po3 = alg3.run()
    chrom3, f1c, f2c, _ = _best_from_pareto(pp3, po3, task_types, problem)
    sla3 = problem.sla_satisfaction_rate(chrom3)
    results["MOPSO+DW-TOPSIS"]    = {"delay": f1c, "cost": f2c, "sla": sla3}
    pareto_for_plot["MOPSO+DW-TOPSIS"] = po3
    hv_histories["MOPSO+DW-TOPSIS"]    = alg3.history["hypervolume"]
    best_chroms["MOPSO+DW-TOPSIS"]     = chrom3
    print(f"   delay={f1c:.4f}s  cost={f2c:.4f}  SLA={sla3:.2%}  [{time.time()-t0:.1f}s]")

    # ── 4. PSO ────────────────────────────────────────────────────────────
    print("\n[4/6] PSO (single-objective)")
    t0 = time.time()
    alg4 = PSO(problem, n_particles=pop_size, n_iter=n_gen, seed=seed)
    chrom4, f1d, f2d = alg4.run()
    sla4 = problem.sla_satisfaction_rate(chrom4)
    results["PSO"]   = {"delay": f1d, "cost": f2d, "sla": sla4}
    best_chroms["PSO"] = chrom4
    print(f"   delay={f1d:.4f}s  cost={f2d:.4f}  SLA={sla4:.2%}  [{time.time()-t0:.1f}s]")

    # ── 5. Genetic Algorithm ──────────────────────────────────────────────
    print("\n[5/6] Genetic Algorithm")
    t0 = time.time()
    alg5 = GeneticAlgo(problem, pop_size=pop_size, n_gen=n_gen, seed=seed)
    chrom5, f1e, f2e = alg5.run()
    sla5 = problem.sla_satisfaction_rate(chrom5)
    results["GA"]   = {"delay": f1e, "cost": f2e, "sla": sla5}
    best_chroms["GA"] = chrom5
    print(f"   delay={f1e:.4f}s  cost={f2e:.4f}  SLA={sla5:.2%}  [{time.time()-t0:.1f}s]")

    # ── 6. Min-Min ────────────────────────────────────────────────────────
    print("\n[6/6] Min-Min")
    t0 = time.time()
    alg6 = MinMin(problem)
    chrom6, f1f, f2f = alg6.run()
    sla6 = problem.sla_satisfaction_rate(chrom6)
    results["Min-Min"]   = {"delay": f1f, "cost": f2f, "sla": sla6}
    best_chroms["Min-Min"] = chrom6
    print(f"   delay={f1f:.4f}s  cost={f2f:.4f}  SLA={sla6:.2%}  [{time.time()-t0:.1f}s]")

    # ── Visualisation ─────────────────────────────────────────────────────
    pfx = f"results/{scenario_name}"

    plot_pareto_fronts(
        pareto_for_plot,
        title=f"Pareto Front – {scenario_name}",
        save_path=f"{pfx}_pareto.png",
    )

    plot_convergence(
        hv_histories,
        title=f"Hypervolume Convergence – {scenario_name}",
        save_path=f"{pfx}_convergence.png",
    )

    plot_bar_comparison(
        results,
        metrics=["delay", "cost", "sla"],
        metric_labels={"delay": "Avg Delay (s)", "cost": "Total Cost (yuan)", "sla": "SLA Satisfaction"},
        title=f"Performance Comparison – {scenario_name}",
        save_path=f"{pfx}_bars.png",
    )

    plot_load_balance(
        {"Imp.NSGA-II+DW-TOPSIS": chrom1, "Min-Min": chrom6, "PSO": chrom4},
        tasks, platforms,
        title=f"Platform Load – {scenario_name}",
        save_path=f"{pfx}_load.png",
    )

    plot_task_node_heatmap(
        {"Imp.NSGA-II+DW-TOPSIS": chrom1, "Min-Min": chrom6},
        tasks, platforms,
        title=f"Task-Node Matching – {scenario_name}",
        save_path=f"{pfx}_heatmap.png",
    )

    _print_table(results, f"SUMMARY – {scenario_name}")
    print(f"\nPlots saved to results/{scenario_name}_*.png")

    return {
        "results": results,
        "pareto": pareto_for_plot,
        "hv_histories": hv_histories,
        "best_chroms": best_chroms,
        "tasks": tasks,
        "platforms": platforms,
    }


# ============================================================
# Scalability experiment
# ============================================================

def run_scalability(
    task_sizes: List[int] = None,
    pop_size: int = 80,
    n_gen: int = 150,
    seed: int = 42,
):
    if task_sizes is None:
        task_sizes = [20, 50, 100, 200]

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for n in task_sizes:
        out = run_single_experiment(
            n_tasks=n, pop_size=pop_size, n_gen=n_gen,
            seed=seed, scenario_name=f"scale_{n}",
        )
        all_results[f"N={n}"] = out["results"]

    plot_scalability(
        task_sizes, all_results,
        save_path="results/scalability.png",
    )
    print("\nScalability plots saved to results/scalability.png")
    return all_results


# ============================================================
# Task-type sensitivity experiment
# ============================================================

def run_sensitivity(pop_size: int = 80, n_gen: int = 150, seed: int = 42):
    scenarios = {
        "delay_sensitive": {
            TaskType.AR_RENDERING:    0.40,
            TaskType.VIDEO_STREAMING: 0.30,
            TaskType.LIGHT_INFERENCE: 0.20,
            TaskType.DATA_ANALYSIS:   0.10,
        },
        "cost_sensitive": {
            TaskType.AI_TRAINING:        0.35,
            TaskType.SCIENTIFIC_COMPUTE: 0.30,
            TaskType.DATA_ANALYSIS:      0.20,
            TaskType.LIGHT_INFERENCE:    0.15,
        },
        "mixed": {
            TaskType.AI_TRAINING:        0.20,
            TaskType.AR_RENDERING:       0.20,
            TaskType.VIDEO_STREAMING:    0.15,
            TaskType.LIGHT_INFERENCE:    0.20,
            TaskType.SCIENTIFIC_COMPUTE: 0.10,
            TaskType.DATA_ANALYSIS:      0.15,
        },
    }
    pref_map = {
        "delay_sensitive": np.array([0.8, 0.2]),
        "cost_sensitive":  np.array([0.2, 0.8]),
        "mixed":           np.array([0.5, 0.5]),
    }

    for name, weights in scenarios.items():
        run_single_experiment(
            n_tasks=50, pop_size=pop_size, n_gen=n_gen,
            preference=pref_map[name], alpha=0.6,
            seed=seed, scenario_name=f"sensitivity_{name}",
        )
