from dataclasses import dataclass

@dataclass
class OptimizationSettings:
    '''
    Hyperparameters for the CVXPY model
    '''
    edge_penalty_weight: float
    size_bonus_weight: float
    tolerance: float = 0

@dataclass
class ImageSettings:
    '''
    Settings for the image tiling
    '''
    num_rows: int
    num_cols: int
    block_size: int