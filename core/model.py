import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from typing import Optional
from .tiles import TileSet, Tile
from .config import OptimizationSettings
from .image_processor import ImageData, Palette

@dataclass
class SolverResult:
    '''
    A result packet from the optimizer
    '''
    is_success: bool
    status: str
    values: Optional[np.ndarray] = None


class TilingOptimizer:
    '''
    Creates a CVXPY problem to tile an image
    '''
    def __init__(self, image_data: ImageData, tile_set: TileSet, settings: OptimizationSettings, palette: Palette):
        self.image_data = image_data
        self.tile_set = tile_set
        self.settings = settings
        self.palette = palette

        self._x = cp.Variable(tile_set.num_placements, boolean=True)

    def _build_constraints(self) -> list[cp.Constraint]:
        constraints = []
        block_to_placements = self.tile_set.block_to_placements()

        for coord, placement_indices in block_to_placements.items():
            constraints.append(cp.sum(self._x[placement_indices]) == 1)
        
        return constraints

    def _build_objective(self) -> cp.Minimize:
        costs = np.zeros(self.tile_set.num_placements)
        edge_weight = self.settings.edge_penalty_weight
        size_weight = self.settings.size_bonus_weight
        colour_to_brightness = self.palette.colour_to_brightness()
        edge_arr = self.image_data.laplacian
        brightness_arr = self.image_data.brightness_arr

        for p, (tile, (i, j)) in enumerate(self.tile_set.placements):
            err = 0
            max_edge = 0
            for (ii, jj) in tile.anchor_footprint((i, j)):
                err += (colour_to_brightness[tile.colour] - brightness_arr[ii,jj])**2
                max_edge = max(max_edge, edge_arr[ii,jj])

            edge_pen = edge_weight * max_edge * (tile.scale - 1)**2
            size_bonus = -size_weight * (tile.scale - 1)

            costs[p] = err + edge_pen + size_bonus

        return cp.Minimize(costs @ self._x)

    def solve(self) -> SolverResult:
        constraints = self._build_constraints()
        objective = self._build_objective()
        problem = cp.Problem(objective, constraints)
        tol = self.settings.tolerance

        print("Solving...")
        problem.solve(verbose=True, solver='HIGHS',
            highs_options={
                "mip_rel_gap": tol,
                })
        
        if problem.status != 'optimal':
            return SolverResult(False, problem.status)
        else:
            return SolverResult(True, problem.status, self._x.value)
