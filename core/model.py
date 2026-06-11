import numpy as np
import cvxpy as cp
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from .tiles import TileSet
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
        self._x = None
        self.costs = []
        self.placements = []
        self.num_placements = 0
        self.block_to_placements = defaultdict(list)

    def prepare(self):
        raw_placements, raw_num_placements = self.tile_set.generate_placements(
            self.image_data.num_cols,
            self.image_data.num_rows
        )
        print(f"Your problem has {raw_num_placements} possible tile placements")
        edge_weight = self.settings.edge_penalty_weight
        size_weight = self.settings.size_bonus_weight
        edge_grid = self.image_data.laplacian_grid
        rgb_grid = self.image_data.rgb_grid
        presolve = self.settings.presolve
        threshold = self.settings.presolve_threshold

        for (tile, (i, j)) in raw_placements:
            err = 0
            max_edge = 0
            normalized_tile = np.array(tile.colour) / 255.0
            anchor_footprint = tile.anchor_footprint((i, j))
            for (ii, jj) in anchor_footprint:
                err += np.sum((normalized_tile - rgb_grid[ii, jj]) ** 2)
                max_edge = max(max_edge, edge_grid[ii,jj])

            edge_pen = edge_weight * max_edge * (tile.scale - 1)**2
            size_bonus = -size_weight * (tile.scale - 1)**2
            tile_cost = err + edge_pen + size_bonus

            if presolve and (tile_cost / len(tile.footprint)) > threshold:
                continue

            self.costs.append(tile_cost)
            self.placements.append((tile, (i, j)))
            for coord in anchor_footprint:
                self.block_to_placements[coord].append(self.num_placements)
            self.num_placements += 1

        self._x = cp.Variable(self.num_placements, boolean=True)
        if presolve:
            print(f"Presolve reduction: placements {self.num_placements}(-{raw_num_placements-self.num_placements})")

    def _build_objective(self) -> cp.Minimize:
        return cp.Minimize(cp.sum(cp.multiply(self.costs, self._x)))
    
    def _build_constraints(self) -> list[cp.Constraint]:
        assert self._x is not None
        constraints = []
        for coord, placement_indices in self.block_to_placements.items():
            constraints.append(cp.sum(self._x[placement_indices]) == 1)
        
        return constraints

    def solve(self) -> SolverResult:
        if self._x is None:
            self.prepare()
            assert self._x is not None

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
