import numpy as np
from typing import List, Tuple

from models import Platform, Task, REGION_DISTANCES


class SchedulingProblem:
    """
    Task-platform matching optimization for Computing Power Internet.

    Decision variable: chromosome[i] = j  means task i is assigned to platform j.

    Objectives (both minimised):
        f1  average task completion delay  (s)
        f2  total scheduling cost          (yuan)

    Constraint:
        Each platform's total workload <= its max_capacity.
        Each task's actual delay       <= its SLA deadline.
    """

    LIGHT_SPEED_KM_S = 2e5  # approximate propagation speed in fibre

    def __init__(self, tasks: List[Task], platforms: List[Platform]):
        self.tasks     = tasks
        self.platforms = platforms
        self.n_tasks    = len(tasks)
        self.n_platforms = len(platforms)
        self._build_matrices()

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    def _build_matrices(self):
        N, M = self.n_tasks, self.n_platforms
        self.delay_matrix = np.zeros((N, M))
        self.cost_matrix  = np.zeros((N, M))

        for i, task in enumerate(self.tasks):
            for j, plat in enumerate(self.platforms):
                dist = REGION_DISTANCES[task.origin_region.value][plat.region.value]

                # --- delay ---
                comp_delay  = task.compute_req / plat.compute_capacity
                prop_delay  = dist / self.LIGHT_SPEED_KM_S
                bw_GBs      = plat.bandwidth / 8.0          # Gbps -> GB/s
                trans_delay = task.data_size / (bw_GBs + 1e-9) + prop_delay
                self.delay_matrix[i, j] = comp_delay + trans_delay

                # --- cost ---
                comp_cost = task.compute_req * plat.cost_per_tflop
                if task.origin_region == plat.region:
                    trans_cost = task.data_size * 0.01
                else:
                    trans_cost = task.data_size * 0.04 * (dist / 1000.0)
                self.cost_matrix[i, j] = comp_cost + trans_cost

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, chromosome: np.ndarray) -> Tuple[float, float, float]:
        """
        Returns (f1_avg_delay, f2_total_cost, constraint_violation).
        Violation > 0 means at least one constraint is broken.
        """
        chrom = chromosome.astype(int)
        delays = self.delay_matrix[np.arange(self.n_tasks), chrom]
        costs  = self.cost_matrix [np.arange(self.n_tasks), chrom]

        f1 = float(np.mean(delays))
        f2 = float(np.sum(costs))

        # capacity constraint
        platform_loads = np.bincount(chrom, weights=[t.compute_req for t in self.tasks],
                                     minlength=self.n_platforms)
        max_caps = np.array([p.max_capacity for p in self.platforms])
        cap_violation = float(np.sum(np.maximum(platform_loads - max_caps, 0.0)))

        # SLA constraint
        deadlines = np.array([t.deadline for t in self.tasks])
        sla_violation = float(np.sum(np.maximum(delays - deadlines, 0.0)))

        violation = cap_violation + sla_violation * 10.0
        return f1, f2, violation

    def evaluate_batch(self, population: np.ndarray) -> np.ndarray:
        """Evaluate all chromosomes. Returns ndarray shape (pop, 3): [f1, f2, viol]."""
        return np.array([self.evaluate(c) for c in population])

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def platform_utilisation(self, chromosome: np.ndarray) -> np.ndarray:
        """Return load-utilisation ratio [0..] for each platform."""
        chrom = chromosome.astype(int)
        loads = np.bincount(chrom, weights=[t.compute_req for t in self.tasks],
                            minlength=self.n_platforms)
        caps = np.array([p.max_capacity for p in self.platforms])
        return loads / (caps + 1e-9)

    def sla_satisfaction_rate(self, chromosome: np.ndarray) -> float:
        """Fraction of tasks that meet their SLA deadline."""
        chrom = chromosome.astype(int)
        delays    = self.delay_matrix[np.arange(self.n_tasks), chrom]
        deadlines = np.array([t.deadline for t in self.tasks])
        return float(np.mean(delays <= deadlines))
