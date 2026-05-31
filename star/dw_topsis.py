"""
DW-TOPSIS: Dual-Weight TOPSIS multi-attribute decision model.

Combines:
  - Subjective weights  (derived from task-type mix, reflecting operational priority)
  - Objective weights   (entropy method, reflecting information content of each indicator)

The combined weight guides selection of the best solution from the NSGA-II Pareto set.
"""

import numpy as np
from typing import Dict, List

from models import TaskType


# Subjective weight templates: [w_delay, w_cost] for each task type
_TASK_W: Dict[TaskType, np.ndarray] = {
    TaskType.AI_TRAINING:        np.array([0.30, 0.70]),  # cost-sensitive
    TaskType.SCIENTIFIC_COMPUTE: np.array([0.40, 0.60]),
    TaskType.AR_RENDERING:       np.array([0.80, 0.20]),  # latency-sensitive
    TaskType.VIDEO_STREAMING:    np.array([0.70, 0.30]),
    TaskType.LIGHT_INFERENCE:    np.array([0.50, 0.50]),
    TaskType.DATA_ANALYSIS:      np.array([0.40, 0.60]),
}


class DWTOPSIS:
    """
    DW-TOPSIS decision model.

    Parameters
    ----------
    task_types : list of TaskType
        The task types of all tasks in the current scheduling batch.
        Used to compute the aggregated subjective weight vector.
    """

    def __init__(self, task_types: List[TaskType]):
        self.task_types = task_types
        self._w_subj = self._aggregate_subjective(task_types)

    # ------------------------------------------------------------------
    # Subjective weights
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_subjective(task_types: List[TaskType]) -> np.ndarray:
        total = np.zeros(2)
        for tt in task_types:
            total += _TASK_W.get(tt, np.array([0.5, 0.5]))
        return total / (total.sum() + 1e-12)

    # ------------------------------------------------------------------
    # Objective weights (entropy method)
    # ------------------------------------------------------------------

    @staticmethod
    def _entropy_weights(matrix: np.ndarray) -> np.ndarray:
        """
        Compute entropy-based objective weights.
        matrix : (n_solutions, n_criteria)  — all values > 0
        """
        n, m = matrix.shape
        col_sum = matrix.sum(axis=0)
        col_sum[col_sum < 1e-12] = 1e-12
        P = matrix / col_sum                    # normalised probability matrix

        eps = 1e-12
        ln_P = np.where(P > eps, np.log(P + eps), 0.0)
        E = -1.0 / np.log(n + eps) * (P * ln_P).sum(axis=0)
        E = np.clip(E, 0.0, 1.0)

        D = 1.0 - E                             # divergence
        w = D / (D.sum() + 1e-12)
        return w

    # ------------------------------------------------------------------
    # Main decision
    # ------------------------------------------------------------------

    def decide(
        self,
        pareto_chromosomes: np.ndarray,
        pareto_objectives: np.ndarray,
    ) -> int:
        """
        Select the index of the best solution in the Pareto set.

        Parameters
        ----------
        pareto_chromosomes : (k, n_tasks)  integer array
        pareto_objectives  : (k, 2)        [avg_delay, total_cost]

        Returns
        -------
        int — row index of the chosen solution in pareto_chromosomes
        """
        if len(pareto_chromosomes) == 1:
            return 0

        mat = pareto_objectives.copy()
        # Shift all values positive (required for entropy)
        mat = mat - mat.min(axis=0) + 1e-6

        # Weights
        w_obj  = self._entropy_weights(mat)
        w_comb = self._w_subj * w_obj
        w_comb /= w_comb.sum() + 1e-12

        # Normalise decision matrix (vector normalisation)
        denom = np.sqrt((mat ** 2).sum(axis=0))
        denom[denom < 1e-12] = 1e-12
        R = mat / denom

        # Weighted normalised matrix
        V = R * w_comb

        # Ideal (+) and negative-ideal (–) solutions (all criteria: minimise)
        A_pos = V.min(axis=0)
        A_neg = V.max(axis=0)

        # Euclidean distances
        S_pos = np.sqrt(((V - A_pos) ** 2).sum(axis=1))
        S_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))

        # Relative closeness (higher = better)
        C = S_neg / (S_pos + S_neg + 1e-12)

        return int(np.argmax(C))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def weight_report(self, pareto_objectives: np.ndarray) -> Dict[str, np.ndarray]:
        """Return all three weight vectors for analysis / logging."""
        mat = pareto_objectives.copy()
        mat = mat - mat.min(axis=0) + 1e-6
        w_obj  = self._entropy_weights(mat)
        w_comb = self._w_subj * w_obj
        w_comb /= w_comb.sum() + 1e-12
        return {
            "subjective": self._w_subj.copy(),
            "objective":  w_obj,
            "combined":   w_comb,
        }
