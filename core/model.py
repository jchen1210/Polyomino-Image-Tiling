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
    placements: Optional[list[tuple]] = None


class TilingOptimizer:
    '''
    Creates a CVXPY problem to tile an image
    '''
    def __init__(self, image_data: ImageData, tile_set: TileSet, settings: OptimizationSettings, palette: Palette):
        self.image_data = image_data
        self.tile_set = tile_set
        self.settings = settings
        self.palette = palette

        self.placements, self.block_to_placements, self.num_placements = tile_set.generate_placements(
            image_data.num_cols,
            image_data.num_rows
        )

        self._x = cp.Variable(self.num_placements, boolean=True)

    def _build_constraints(self) -> list[cp.Constraint]:
        constraints = []
        block_to_placements = self.block_to_placements

        for coord, placement_indices in block_to_placements.items():
            constraints.append(cp.sum(self._x[placement_indices]) == 1)
        
        return constraints

    def _build_objective(self) -> cp.Minimize:
        costs = np.zeros(self.num_placements)
        edge_weight = self.settings.edge_penalty_weight
        size_weight = self.settings.size_bonus_weight
        edge_grid = self.image_data.laplacian_grid
        rgb_grid = self.image_data.rgb_grid

        for p, (tile, (i, j)) in enumerate(self.placements):
            err = 0
            max_edge = 0
            normalized_tile = np.array(tile.colour) / 255.0
            for (ii, jj) in tile.anchor_footprint((i, j)):
                err += np.sum((normalized_tile - rgb_grid[ii, jj]) ** 2)
                max_edge = max(max_edge, edge_grid[ii,jj])

            edge_pen = edge_weight * max_edge * (tile.scale - 1)**2
            size_bonus = -size_weight * (tile.scale - 1)**2

            costs[p] = err + edge_pen + size_bonus

        return cp.Minimize(cp.sum(cp.multiply(costs, self._x)))

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
            return SolverResult(True, problem.status, self._x.value, self.placements)
