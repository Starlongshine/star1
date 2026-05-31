"""
Synthetic data generator for Computing Power Internet scheduling experiments.

Generates platforms (core / regional / edge nodes) and tasks
(AI training, scientific compute, AR rendering, video streaming,
 light inference, data analysis) with realistic parameter ranges.
"""

import numpy as np
from typing import Dict, List, Tuple

from models import NodeType, Platform, Region, Task, TaskType


# ---------------------------------------------------------------------------
# Platform generation
# ---------------------------------------------------------------------------

def generate_platforms(
    n_core: int = 3,
    n_regional: int = 4,
    n_edge: int = 5,
    seed: int = 42,
) -> List[Platform]:
    np.random.seed(seed)
    platforms: List[Platform] = []
    regions = list(Region)
    pid = 0

    for i in range(n_core):
        r = regions[i % len(regions)]
        platforms.append(Platform(
            id=pid, name=f"Core-{r.name}-{i+1}",
            node_type=NodeType.CORE, region=r,
            compute_capacity=np.random.uniform(100, 200),
            bandwidth=np.random.uniform(20, 40),
            energy_rate=np.random.uniform(0.5, 1.0),
            cost_per_tflop=np.random.uniform(0.06, 0.12),
            max_capacity=np.random.uniform(6000, 12000),
            availability=np.random.uniform(0.98, 0.999),
        ))
        pid += 1

    for i in range(n_regional):
        r = regions[i % len(regions)]
        platforms.append(Platform(
            id=pid, name=f"Regional-{r.name}-{i+1}",
            node_type=NodeType.REGIONAL, region=r,
            compute_capacity=np.random.uniform(30, 80),
            bandwidth=np.random.uniform(8, 20),
            energy_rate=np.random.uniform(0.3, 0.6),
            cost_per_tflop=np.random.uniform(0.025, 0.06),
            max_capacity=np.random.uniform(2000, 5000),
            availability=np.random.uniform(0.95, 0.99),
        ))
        pid += 1

    for i in range(n_edge):
        r = regions[i % len(regions)]
        platforms.append(Platform(
            id=pid, name=f"Edge-{r.name}-{i+1}",
            node_type=NodeType.EDGE, region=r,
            compute_capacity=np.random.uniform(5, 20),
            bandwidth=np.random.uniform(2, 10),
            energy_rate=np.random.uniform(0.1, 0.3),
            cost_per_tflop=np.random.uniform(0.005, 0.02),
            max_capacity=np.random.uniform(300, 1000),
            availability=np.random.uniform(0.90, 0.97),
        ))
        pid += 1

    return platforms


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: Dict[TaskType, float] = {
    TaskType.AI_TRAINING:        0.20,
    TaskType.SCIENTIFIC_COMPUTE: 0.15,
    TaskType.AR_RENDERING:       0.20,
    TaskType.VIDEO_STREAMING:    0.15,
    TaskType.LIGHT_INFERENCE:    0.20,
    TaskType.DATA_ANALYSIS:      0.10,
}

# Parameter ranges per task type:
#   (compute_req_range, data_size_range, deadline_range, priority_range)
_TASK_PARAMS: Dict[TaskType, Tuple] = {
    TaskType.AI_TRAINING:        ((50,  800), (5,  80),  (20,  300), (3, 5)),
    TaskType.SCIENTIFIC_COMPUTE: ((40,  600), (3,  60),  (15,  200), (2, 5)),
    TaskType.AR_RENDERING:       ((0.5,  8),  (0.05, 2), (0.5,  5),  (4, 5)),
    TaskType.VIDEO_STREAMING:    ((1,   15),  (0.1,  5), (1,    8),  (3, 5)),
    TaskType.LIGHT_INFERENCE:    ((0.2,  5),  (0.01, 1), (1,   30),  (1, 3)),
    TaskType.DATA_ANALYSIS:      ((1,   20),  (0.5, 10), (5,   60),  (1, 4)),
}


def generate_tasks(
    n_tasks: int = 50,
    task_type_weights: Dict[TaskType, float] = None,
    seed: int = 42,
) -> List[Task]:
    np.random.seed(seed)
    if task_type_weights is None:
        task_type_weights = _DEFAULT_WEIGHTS

    types   = list(task_type_weights.keys())
    weights = np.array([task_type_weights[t] for t in types], dtype=float)
    weights /= weights.sum()

    regions = list(Region)
    tasks: List[Task] = []

    for i in range(n_tasks):
        tt: TaskType = np.random.choice(types, p=weights)  # type: ignore
        cr, ds, dl, pr = _TASK_PARAMS[tt]
        tasks.append(Task(
            id=i,
            task_type=tt,
            compute_req=np.random.uniform(*cr),
            data_size=np.random.uniform(*ds),
            deadline=np.random.uniform(*dl),
            priority=int(np.random.randint(pr[0], pr[1]+1)),
            origin_region=np.random.choice(regions),
        ))

    return tasks


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def generate_scenario(
    n_tasks: int = 50,
    n_core: int = 3,
    n_regional: int = 4,
    n_edge: int = 5,
    task_type_weights: Dict[TaskType, float] = None,
    seed: int = 42,
) -> Tuple[List[Task], List[Platform]]:
    platforms = generate_platforms(n_core, n_regional, n_edge, seed)
    tasks     = generate_tasks(n_tasks, task_type_weights, seed)
    return tasks, platforms
