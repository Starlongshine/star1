"""
Baseline scheduling algorithms for comparison.

Implemented:
  MinMin        – classic heuristic
  PSO           – particle swarm optimisation (single-objective)
  GeneticAlgo   – standard genetic algorithm  (single-objective)
  StandardNSGAII– NSGA-II without any improvements
  MOPSO         – multi-objective PSO with external Pareto archive
"""

import numpy as np
from typing import List, Tuple

from problem import SchedulingProblem


# ============================================================
# Min-Min
# ============================================================

class MinMin:
    """
    Classic Min-Min heuristic:
    Repeatedly pick the unscheduled task whose minimum completion time
    (across all platforms) is smallest, and assign it there.
    """

    def __init__(self, problem: SchedulingProblem):
        self.problem = problem

    def run(self) -> Tuple[np.ndarray, float, float]:
        N, M = self.problem.n_tasks, self.problem.n_platforms
        remaining   = list(range(N))
        chromosome  = np.zeros(N, dtype=int)
        plat_finish = np.zeros(M)   # earliest-available finish time per platform

        while remaining:
            best_task, best_plat, best_finish = None, None, np.inf
            for i in remaining:
                for j in range(M):
                    cap = self.problem.platforms[j].compute_capacity
                    finish = plat_finish[j] + self.problem.delay_matrix[i, j]
                    if finish < best_finish:
                        best_finish = finish
                        best_task   = i
                        best_plat   = j

            chromosome[best_task] = best_plat
            cap_j = self.problem.platforms[best_plat].compute_capacity
            plat_finish[best_plat] += self.problem.delay_matrix[best_task, best_plat]
            remaining.remove(best_task)

        f1, f2, _ = self.problem.evaluate(chromosome)
        return chromosome, f1, f2


# ============================================================
# PSO  (single-objective: minimise avg delay)
# ============================================================

class PSO:

    def __init__(
        self,
        problem: SchedulingProblem,
        n_particles: int = 100,
        n_iter: int = 200,
        w: float = 0.72,
        c1: float = 1.49,
        c2: float = 1.49,
        seed: int = 42,
    ):
        self.problem     = problem
        self.n_particles = n_particles
        self.n_iter      = n_iter
        self.w, self.c1, self.c2 = w, c1, c2
        np.random.seed(seed)
        self.N = problem.n_tasks
        self.M = problem.n_platforms

    def _to_chrom(self, pos: np.ndarray) -> np.ndarray:
        return np.clip(pos, 0, self.M - 1e-6).astype(int)

    def _fitness(self, pos: np.ndarray) -> float:
        c = self._to_chrom(pos)
        f1, f2, v = self.problem.evaluate(c)
        return f1 + 1000.0 * v

    def run(self) -> Tuple[np.ndarray, float, float]:
        pos = np.random.uniform(0, self.M, (self.n_particles, self.N))
        vel = np.random.uniform(-self.M * 0.2, self.M * 0.2, (self.n_particles, self.N))

        pbest     = pos.copy()
        pbest_fit = np.array([self._fitness(p) for p in pos])
        g         = int(np.argmin(pbest_fit))
        gbest     = pbest[g].copy()
        gbest_fit = pbest_fit[g]

        for _ in range(self.n_iter):
            r1 = np.random.random((self.n_particles, self.N))
            r2 = np.random.random((self.n_particles, self.N))
            vel = self.w * vel + self.c1 * r1 * (pbest - pos) + self.c2 * r2 * (gbest - pos)
            pos = np.clip(pos + vel, 0, self.M - 1e-6)

            for i in range(self.n_particles):
                f = self._fitness(pos[i])
                if f < pbest_fit[i]:
                    pbest[i]     = pos[i].copy()
                    pbest_fit[i] = f
                    if f < gbest_fit:
                        gbest     = pos[i].copy()
                        gbest_fit = f

        chrom = self._to_chrom(gbest)
        f1, f2, _ = self.problem.evaluate(chrom)
        return chrom, f1, f2


# ============================================================
# Genetic Algorithm (single-objective)
# ============================================================

class GeneticAlgo:

    def __init__(
        self,
        problem: SchedulingProblem,
        pop_size: int = 100,
        n_gen: int = 200,
        pm: float = 0.10,
        pc: float = 0.90,
        seed: int = 42,
    ):
        self.problem  = problem
        self.pop_size = pop_size
        self.n_gen    = n_gen
        self.pm, self.pc = pm, pc
        np.random.seed(seed)
        self.N = problem.n_tasks
        self.M = problem.n_platforms

    def _fitness(self, c: np.ndarray) -> float:
        f1, f2, v = self.problem.evaluate(c)
        return f1 + 0.01 * f2 + 1000.0 * v

    def run(self) -> Tuple[np.ndarray, float, float]:
        pop  = np.random.randint(0, self.M, (self.pop_size, self.N))
        fits = np.array([self._fitness(c) for c in pop])

        for _ in range(self.n_gen):
            # tournament selection
            new_pop = np.empty_like(pop)
            for i in range(self.pop_size):
                a, b = np.random.choice(self.pop_size, 2, replace=False)
                new_pop[i] = pop[a] if fits[a] <= fits[b] else pop[b]

            # crossover
            for i in range(0, self.pop_size - 1, 2):
                if np.random.random() < self.pc:
                    pt = np.random.randint(1, self.N)
                    new_pop[i, pt:], new_pop[i+1, pt:] = (
                        new_pop[i+1, pt:].copy(),
                        new_pop[i,   pt:].copy(),
                    )

            # mutation
            mask = np.random.random((self.pop_size, self.N)) < self.pm
            new_pop[mask] = np.random.randint(0, self.M, int(mask.sum()))

            # elitism: keep best
            best_idx = int(np.argmin(fits))
            worst_new = int(np.argmax([self._fitness(c) for c in new_pop]))
            new_pop[worst_new] = pop[best_idx]

            pop  = new_pop
            fits = np.array([self._fitness(c) for c in pop])

        best = pop[int(np.argmin(fits))]
        f1, f2, _ = self.problem.evaluate(best)
        return best, f1, f2


# ============================================================
# Standard NSGA-II  (no improvements, for ablation)
# ============================================================

class StandardNSGAII:

    def __init__(
        self,
        problem: SchedulingProblem,
        pop_size: int = 100,
        n_gen: int = 200,
        pm: float = 0.10,
        pc: float = 0.90,
        penalty_coeff: float = 1000.0,
        seed: int = 42,
    ):
        self.problem  = problem
        self.pop_size = pop_size
        self.n_gen    = n_gen
        self.pm, self.pc = pm, pc
        self.penalty_coeff = penalty_coeff
        np.random.seed(seed)
        self.N = problem.n_tasks
        self.M = problem.n_platforms
        self.history = {"hypervolume": []}

    # --- helpers ---
    @staticmethod
    def _dom(a, b):
        return bool(np.all(a <= b) and np.any(a < b))

    def _nds(self, obj):
        n = len(obj)
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        fronts = [[]]
        for i in range(n):
            for j in range(i+1, n):
                if self._dom(obj[i], obj[j]):  S[i].append(j);  nd[j] += 1
                elif self._dom(obj[j], obj[i]): S[j].append(i); nd[i] += 1
            if nd[i] == 0:
                fronts[0].append(i)
        k = 0
        while fronts[k]:
            nxt = []
            for i in fronts[k]:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0: nxt.append(j)
            fronts.append(nxt);  k += 1
        return [f for f in fronts if f]

    def _cd(self, front, obj):
        n = len(front)
        if n <= 2: return np.full(n, np.inf)
        fo = obj[front]
        d = np.zeros(n)
        for m in range(fo.shape[1]):
            si  = np.argsort(fo[:, m])
            rng = fo[si[-1], m] - fo[si[0], m]
            if rng < 1e-12: continue
            d[si[0]] = d[si[-1]] = np.inf
            for k in range(1, n-1):
                d[si[k]] += (fo[si[k+1], m] - fo[si[k-1], m]) / rng
        return d

    def _pen(self, raw):
        return raw[:, :2] + self.penalty_coeff * raw[:, [2, 2]]

    def _hv(self, pf, ref):
        mask = (pf[:, 0] < ref[0]) & (pf[:, 1] < ref[1])
        pts  = pf[mask]
        if len(pts) == 0: return 0.0
        pts = pts[np.argsort(pts[:, 0])]
        hv, prev_f2 = 0.0, ref[1]
        for p in pts:
            if p[1] < prev_f2:
                hv += (ref[0] - p[0]) * (prev_f2 - p[1])
                prev_f2 = p[1]
        return hv

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        pop = np.random.randint(0, self.M, (self.pop_size, self.N))
        obj = self._pen(self.problem.evaluate_batch(pop))

        for gen in range(self.n_gen):
            fronts = self._nds(obj)
            ranks  = np.zeros(self.pop_size, dtype=int)
            crowd  = np.zeros(self.pop_size)
            for r, f in enumerate(fronts):
                for i in f: ranks[i] = r
                cd = self._cd(f, obj)
                for k, i in enumerate(f): crowd[i] = cd[k]

            # reproduce
            offspring = np.zeros_like(pop)
            for i in range(0, self.pop_size, 2):
                def tour():
                    a, b = np.random.choice(self.pop_size, 2, replace=False)
                    if ranks[a] < ranks[b]: return a
                    if ranks[b] < ranks[a]: return b
                    return a if crowd[a] >= crowd[b] else b
                p1, p2 = tour(), tour()
                if np.random.random() < self.pc:
                    pt = np.random.randint(1, self.N)
                    c1 = np.concatenate([pop[p1,:pt], pop[p2,pt:]])
                    c2 = np.concatenate([pop[p2,:pt], pop[p1,pt:]])
                else:
                    c1, c2 = pop[p1].copy(), pop[p2].copy()
                m1 = np.random.random(self.N) < self.pm
                m2 = np.random.random(self.N) < self.pm
                c1[m1] = np.random.randint(0, self.M, m1.sum())
                c2[m2] = np.random.randint(0, self.M, m2.sum())
                offspring[i] = c1
                if i+1 < self.pop_size: offspring[i+1] = c2

            off_obj  = self._pen(self.problem.evaluate_batch(offspring))
            comb_pop = np.vstack([pop, offspring])
            comb_obj = np.vstack([obj, off_obj])

            fronts2  = self._nds(comb_obj)
            selected: List[int] = []
            for f in fronts2:
                if len(selected) + len(f) <= self.pop_size:
                    selected.extend(f)
                else:
                    needed = self.pop_size - len(selected)
                    cd2 = self._cd(f, comb_obj)
                    order = sorted(range(len(f)), key=lambda k: -cd2[k])
                    selected.extend(f[i] for i in order[:needed])
                    break

            sel = np.array(selected)
            pop, obj = comb_pop[sel], comb_obj[sel]

            pf_obj = obj[self._nds(obj)[0]]
            ref = obj.max(axis=0) * 1.1
            self.history["hypervolume"].append(self._hv(pf_obj, ref))

        pf = self._nds(obj)[0]
        return pop[pf], obj[pf]


# ============================================================
# MOPSO  (multi-objective PSO with external archive)
# ============================================================

class MOPSO:

    def __init__(
        self,
        problem: SchedulingProblem,
        n_particles: int = 100,
        n_iter: int = 200,
        w: float = 0.72,
        c1: float = 1.49,
        c2: float = 1.49,
        archive_size: int = 100,
        penalty_coeff: float = 1000.0,
        seed: int = 42,
    ):
        self.problem      = problem
        self.n_particles  = n_particles
        self.n_iter       = n_iter
        self.w, self.c1, self.c2 = w, c1, c2
        self.archive_size = archive_size
        self.penalty_coeff = penalty_coeff
        np.random.seed(seed)
        self.N = problem.n_tasks
        self.M = problem.n_platforms
        self.history = {"hypervolume": []}

    @staticmethod
    def _dom(a, b):
        return bool(np.all(a <= b) and np.any(a < b))

    def _eval(self, pos: np.ndarray) -> np.ndarray:
        c = np.clip(pos, 0, self.M - 1e-6).astype(int)
        f1, f2, v = self.problem.evaluate(c)
        return np.array([f1 + self.penalty_coeff * v, f2 + self.penalty_coeff * v])

    def _update_archive(self, arch_pos, arch_obj, new_pos, new_obj):
        # check if new_obj is dominated by archive
        for ao in arch_obj:
            if self._dom(ao, new_obj):
                return arch_pos, arch_obj
        # remove dominated by new_obj
        keep = [i for i, ao in enumerate(arch_obj) if not self._dom(new_obj, ao)]
        arch_pos = [arch_pos[i] for i in keep]
        arch_obj = [arch_obj[i] for i in keep]
        arch_pos.append(new_pos.copy())
        arch_obj.append(new_obj.copy())
        if len(arch_pos) > self.archive_size:
            drop = np.random.randint(len(arch_pos))
            arch_pos.pop(drop)
            arch_obj.pop(drop)
        return arch_pos, arch_obj

    def _hv(self, obj_arr, ref):
        if len(obj_arr) == 0: return 0.0
        pts = obj_arr[(obj_arr[:, 0] < ref[0]) & (obj_arr[:, 1] < ref[1])]
        if len(pts) == 0: return 0.0
        pts = pts[np.argsort(pts[:, 0])]
        hv, prev = 0.0, ref[1]
        for p in pts:
            if p[1] < prev:
                hv += (ref[0] - p[0]) * (prev - p[1])
                prev = p[1]
        return hv

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.random.uniform(0, self.M, (self.n_particles, self.N))
        vel = np.zeros_like(pos)

        pbest     = pos.copy()
        pbest_obj = np.array([self._eval(p) for p in pos])

        arch_pos: list = []
        arch_obj: list = []
        for i in range(self.n_particles):
            arch_pos, arch_obj = self._update_archive(arch_pos, arch_obj, pos[i], pbest_obj[i])

        for _ in range(self.n_iter):
            for i in range(self.n_particles):
                leader = arch_pos[np.random.randint(len(arch_pos))] if arch_pos else pbest[i]
                r1, r2 = np.random.random(self.N), np.random.random(self.N)
                vel[i] = self.w * vel[i] + self.c1*r1*(pbest[i]-pos[i]) + self.c2*r2*(leader-pos[i])
                pos[i] = np.clip(pos[i] + vel[i], 0, self.M - 1e-6)
                no = self._eval(pos[i])
                if self._dom(no, pbest_obj[i]):
                    pbest[i]     = pos[i].copy()
                    pbest_obj[i] = no
                arch_pos, arch_obj = self._update_archive(arch_pos, arch_obj, pos[i], no)

            if arch_obj:
                ao = np.array(arch_obj)
                ref = ao.max(axis=0) * 1.1
                self.history["hypervolume"].append(self._hv(ao, ref))
            else:
                self.history["hypervolume"].append(0.0)

        if arch_pos:
            chroms = np.array([np.clip(p, 0, self.M-1e-6).astype(int) for p in arch_pos])
            return chroms, np.array(arch_obj)
        chrom = np.clip(pos[0], 0, self.M-1e-6).astype(int)
        f1, f2, _ = self.problem.evaluate(chrom)
        return np.array([chrom]), np.array([[f1, f2]])
