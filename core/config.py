from dataclasses import dataclass
from typing import Optional
from core import Polyomino

@dataclass
class OptimizationSettings:
    '''
    Hyperparameters for the CVXPY model
    '''
    edge_penalty_weight: float
    size_bonus_weight: float
    tolerance: float = 0
    presolve: bool = True
    presolve_threshold: float = 0.25

@dataclass
class ImageSettings:
    '''
    Settings for the image processor
    '''
    num_rows: int
    num_cols: int
    block_size: int
    auto_pick: bool = True
    requested_colour_budget: Optional[int] = None

@dataclass
class TileConfig:
    '''
    Config for the tiles used to cover an image
    '''
    shapes: list[Polyomino]
    scales: list[int]