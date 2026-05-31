import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Region-to-region propagation distances (km, approximate China geography)
REGION_DISTANCES = np.array([
    #  N     S     E     W     C
    [   0, 2100, 1200, 1800, 1050],  # NORTH  (Beijing)
    [2100,    0, 1200, 1500,  980],  # SOUTH  (Guangzhou)
    [1200, 1200,    0, 2200,  830],  # EAST   (Shanghai)
    [1800, 1500, 2200,    0, 1500],  # WEST   (Chengdu/Xi'an)
    [1050,  980,  830, 1500,    0],  # CENTRAL (Wuhan)
], dtype=float)


class NodeType(Enum):
    CORE     = 0  # high compute, higher cost
    REGIONAL = 1  # mid-tier
    EDGE     = 2  # low latency, lower compute


class TaskType(Enum):
    AI_TRAINING        = 0  # large compute, prefers CORE
    SCIENTIFIC_COMPUTE = 1  # large compute, prefers CORE
    AR_RENDERING       = 2  # strict latency, prefers EDGE
    VIDEO_STREAMING    = 3  # strict latency, prefers EDGE
    LIGHT_INFERENCE    = 4  # flexible
    DATA_ANALYSIS      = 5  # flexible


class Region(Enum):
    NORTH   = 0
    SOUTH   = 1
    EAST    = 2
    WEST    = 3
    CENTRAL = 4


@dataclass
class Platform:
    """One computing-power platform node in the interconnected network."""
    id: int
    name: str
    node_type: NodeType
    region: Region
    compute_capacity: float  # TFLOPS  (capacity per second)
    bandwidth: float         # Gbps    (outgoing network bandwidth)
    energy_rate: float       # W / TFLOP processed
    cost_per_tflop: float    # yuan / TFLOP
    max_capacity: float      # TFLOP   (total workload ceiling)
    availability: float = 0.99


@dataclass
class Task:
    """A computing task submitted to the scheduling platform."""
    id: int
    task_type: TaskType
    compute_req: float    # TFLOP  (total floating-point work needed)
    data_size: float      # GB     (input data volume)
    deadline: float       # s      (SLA deadline)
    priority: int         # 1–5, higher = more important
    origin_region: Region

    @property
    def preferred_node_type(self) -> Optional[NodeType]:
        if self.task_type in (TaskType.AI_TRAINING, TaskType.SCIENTIFIC_COMPUTE):
            return NodeType.CORE
        if self.task_type in (TaskType.AR_RENDERING, TaskType.VIDEO_STREAMING):
            return NodeType.EDGE
        return None
