"""
Improved NSGA-II for Computing Power Internet task scheduling.

Innovations over standard NSGA-II:
  1. Chaotic population initialisation  (logistic map, avoids clustering)
  2. Chaotic adaptive mutation rate     (dynamic pm via logistic sequence)
  3. Direction-guided crowding distance (cosine similarity with preference vector)
"""

import numpy as np
from typing import List, Optional, Tuple

from problem import SchedulingProblem


# ---------------------------------------------------------------------------
# Chaotic utilities
# ---------------------------------------------------------------------------

def _logistic(x: float, mu: float = 4.0) -> float:
    return mu * x * (1.0 - x)


def _chaotic_seq(length: int, x0: float, warm_up: int = 200) -> np.ndarray:
    """Logistic-map chaotic sequence with transient removal."""
    x = x0
    for _ in range(warm_up):
        x = _logistic(x)
    seq = np.empty(length)
    for i in range(length):
        x = _logistic(x)
        seq[i] = x
    return seq


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ImprovedNSGAII:

    def __init__(
        self,
        problem: SchedulingProblem,
        pop_size: int = 100,
        n_generations: int = 200,
        pm_base: float = 0.05,
        pm_max: float = 0.30,
        pc: float = 0.90,
        alpha: float = 0.50,
        preference: Optional[np.ndarray] = None,
        penalty_coeff: float = 1000.0,
        seed: int = 42,
    ):
        self.problem        = problem
        self.pop_size       = pop_size
        self.n_gen          = n_generations
        self.pm_base        = pm_base
        self.pm_max         = pm_max
        self.pc             = pc
        self.alpha          = alpha
        self.preference     = preference if preference is not None else np.array([0.5, 0.5])
        self.penalty_coeff  = penalty_coeff
        self.seed           = seed

        np.random.seed(seed)
        self.N = problem.n_tasks
        self.M = problem.n_platforms

        # Pre-generate chaotic sequence for mutation rate schedule
        self._chaos_pm = _chaotic_seq(n_generations + 10, x0=0.31)

        self.history = {"hypervolume": [], "min_f1": [], "min_f2": []}

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _chaotic_init(self) -> np.ndarray:
        pop = np.zeros((self.pop_size, self.N), dtype=int)
        for i in range(self.pop_size):
            x0 = (i + 1) / (self.pop_size + 2)
            seq = _chaotic_seq(self.N, x0=x0)
            pop[i] = (seq * self.M).astype(int) % self.M
        return pop

    # -----------------------------------------------------------------------
    # Dominance & sorting
    # -----------------------------------------------------------------------

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.all(a <= b) and np.any(a < b))

    def _fast_nds(self, obj: np.ndarray) -> List[List[int]]:
        """Fast non-dominated sorting. Returns list of fronts."""
        n = len(obj)
        S = [[] for _ in range(n)]     # dominated-by sets
        n_dom = np.zeros(n, dtype=int) # domination counters
        fronts: List[List[int]] = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(obj[i], obj[j]):
                    S[i].append(j);  n_dom[j] += 1
                elif self._dominates(obj[j], obj[i]):
                    S[j].append(i);  n_dom[i] += 1
            if n_dom[i] == 0:
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            nxt = []
            for i in fronts[k]:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        nxt.append(j)
            fronts.append(nxt)
            k += 1

        return [f for f in fronts if f]

    # -----------------------------------------------------------------------
    # Direction-guided crowding distance
    # -----------------------------------------------------------------------

    def _dg_crowding(
        self,
        front: List[int],
        obj: np.ndarray,
    ) -> np.ndarray:
        """
        Improved crowding distance:
            CD_imp = alpha * cos_sim_norm + (1-alpha) * std_cd_norm

        The cosine similarity is measured between each solution's direction
        toward the ideal point and the user-specified preference vector.
        Higher value -> the solution aligns better with preferences.
        """
        n = len(front)
        if n <= 2:
            return np.full(n, np.inf)

        fo = obj[front]  # (n, 2)

        # --- standard crowding distance ---
        std = np.zeros(n)
        for m in range(fo.shape[1]):
            idx_s = np.argsort(fo[:, m])
            rng = fo[idx_s[-1], m] - fo[idx_s[0], m]
            if rng < 1e-12:
                continue
            std[idx_s[0]]  = np.inf
            std[idx_s[-1]] = np.inf
            for k in range(1, n - 1):
                std[idx_s[k]] += (fo[idx_s[k+1], m] - fo[idx_s[k-1], m]) / rng

        # --- preference-guided cosine similarity ---
        obj_min = fo.min(axis=0)
        obj_max = fo.max(axis=0)
        rng_vec = obj_max - obj_min
        rng_vec[rng_vec < 1e-12] = 1.0
        norm_fo = (fo - obj_min) / rng_vec          # normalise to [0,1]

        # direction vectors: from each solution toward the ideal (0,0)
        dirs = -norm_fo                              # ideal direction = decrease all
        pref = self.preference / (np.linalg.norm(self.preference) + 1e-12)

        cos = np.zeros(n)
        for k in range(n):
            d = dirs[k]
            dn = np.linalg.norm(d)
            if dn > 1e-12:
                cos[k] = np.dot(d / dn, pref)

        cos_norm = (cos + 1.0) / 2.0   # shift [-1,1] -> [0,1]

        # normalise std (replace inf temporarily)
        finite = np.isfinite(std)
        if finite.any():
            max_std = std[finite].max() if finite.any() else 1.0
        else:
            max_std = 1.0
        std_norm = std.copy()
        std_norm[~finite] = max_std * 2.0
        std_norm /= std_norm.max() + 1e-12

        return self.alpha * cos_norm + (1 - self.alpha) * std_norm

    # -----------------------------------------------------------------------
    # Selection / variation
    # -----------------------------------------------------------------------

    def _tournament(self, ranks: np.ndarray, crowd: np.ndarray) -> int:
        a, b = np.random.choice(len(ranks), 2, replace=False)
        if ranks[a] < ranks[b]:   return a
        if ranks[b] < ranks[a]:   return b
        return a if crowd[a] >= crowd[b] else b

    def _crossover(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > self.pc:
            return p1.copy(), p2.copy()
        pt = np.random.randint(1, self.N)
        c1 = np.concatenate([p1[:pt], p2[pt:]])
        c2 = np.concatenate([p2[:pt], p1[pt:]])
        return c1, c2

    def _mutate(self, ind: np.ndarray, gen: int) -> np.ndarray:
        pm = self.pm_base + (self.pm_max - self.pm_base) * self._chaos_pm[gen]
        mask = np.random.random(self.N) < pm
        out = ind.copy()
        out[mask] = np.random.randint(0, self.M, mask.sum())
        return out

    # -----------------------------------------------------------------------
    # Penalised objectives
    # -----------------------------------------------------------------------

    def _penalise(self, raw: np.ndarray) -> np.ndarray:
        """raw shape (pop, 3) -> penalised objectives (pop, 2)."""
        return raw[:, :2] + self.penalty_coeff * raw[:, [2, 2]]

    # -----------------------------------------------------------------------
    # Environmental selection
    # -----------------------------------------------------------------------

    def _env_select(
        self,
        comb_pop: np.ndarray,
        comb_obj: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        fronts = self._fast_nds(comb_obj)
        n_total = len(comb_pop)
        ranks  = np.zeros(n_total, dtype=int)
        crowd  = np.zeros(n_total)

        for r, f in enumerate(fronts):
            for idx in f:
                ranks[idx] = r

        selected: List[int] = []
        for r, f in enumerate(fronts):
            if len(selected) + len(f) <= self.pop_size:
                selected.extend(f)
            else:
                needed = self.pop_size - len(selected)
                cd = self._dg_crowding(f, comb_obj)
                for k, idx in enumerate(f):
                    crowd[idx] = cd[k]
                order = sorted(range(len(f)), key=lambda k: -crowd[f[k]])
                selected.extend(f[i] for i in order[:needed])
                break

        sel = np.array(selected)
        return comb_pop[sel], comb_obj[sel], ranks[sel], crowd[sel]

    # -----------------------------------------------------------------------
    # Hypervolume (2-D exact sweep)
    # -----------------------------------------------------------------------

    def _hypervolume(self, obj: np.ndarray, ref: np.ndarray) -> float:
        mask = (obj[:, 0] < ref[0]) & (obj[:, 1] < ref[1])
        pts  = obj[mask]
        if len(pts) == 0:
            return 0.0
        pts = pts[np.argsort(pts[:, 0])]
        # remove dominated points (f2 should be decreasing)
        min_f2 = np.inf
        keep = []
        for p in pts[::-1]:
            if p[1] < min_f2:
                keep.append(p)
                min_f2 = p[1]
        pts = np.array(keep[::-1])

        hv, n = 0.0, len(pts)
        for i, p in enumerate(pts):
            nxt_f1 = pts[i+1, 0] if i < n-1 else ref[0]
            hv += (nxt_f1 - p[0]) * (ref[1] - p[1])
        return hv

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run improved NSGA-II.
        Returns (pareto_chromosomes, pareto_objectives) where
        pareto_objectives has shape (k, 2): [avg_delay, total_cost].
        """
        pop = self._chaotic_init()
        raw = self.problem.evaluate_batch(pop)
        obj = self._penalise(raw)

        fronts = self._fast_nds(obj)
        ranks  = np.zeros(self.pop_size, dtype=int)
        crowd  = np.zeros(self.pop_size)
        for r, f in enumerate(fronts):
            for idx in f:
                ranks[idx] = r
            cd = self._dg_crowding(f, obj)
            for k, idx in enumerate(f):
                crowd[idx] = cd[k]

        for gen in range(self.n_gen):
            # --- reproduce ---
            offspring = np.zeros_like(pop)
            for i in range(0, self.pop_size, 2):
                p1 = self._tournament(ranks, crowd)
                p2 = self._tournament(ranks, crowd)
                c1, c2 = self._crossover(pop[p1], pop[p2])
                offspring[i]   = self._mutate(c1, gen)
                if i + 1 < self.pop_size:
                    offspring[i+1] = self._mutate(c2, gen)

            # --- evaluate offspring ---
            off_raw = self.problem.evaluate_batch(offspring)
            off_obj = self._penalise(off_raw)

            # --- environmental selection ---
            comb_pop = np.vstack([pop, offspring])
            comb_obj = np.vstack([obj, off_obj])
            pop, obj, ranks, crowd = self._env_select(comb_pop, comb_obj)

            # --- history ---
            pf_idx = self._fast_nds(obj)[0]
            pf_obj = obj[pf_idx]
            ref = obj.max(axis=0) * 1.1
            self.history["hypervolume"].append(self._hypervolume(pf_obj, ref))
            self.history["min_f1"].append(float(pf_obj[:, 0].min()))
            self.history["min_f2"].append(float(pf_obj[:, 1].min()))

        # --- return Pareto front ---
        pf_idx = self._fast_nds(obj)[0]
        return pop[pf_idx], obj[pf_idx]
