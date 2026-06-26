import numpy as np
import cvxpy as cp
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from core import TileSet, OptimizationSettings, TileConfig, ImageData

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
    def __init__(self, image_data: ImageData, tile_config: TileConfig, settings: OptimizationSettings):
        self.image_data = image_data
        self.settings = settings
        self.tile_set = TileSet(tile_config.shapes, tile_config.scales, image_data.palette)

        self._x = None
        self.costs = None
        self.placements = None
        self.block_to_placements = None        

    def prepare(self):
        t0 = time.monotonic()
        print("Setting up problem...")
        raw_placements, raw_num_placements, raw_costs = self._build_costs()

        warm_start = None
        if self.settings.presolve:
            self.placements, num_placements, self.costs, warm_start = self.presolve(raw_placements, raw_costs)
        else:
            self.placements, num_placements, self.costs = raw_placements, raw_num_placements, raw_costs

        t1 = time.monotonic()
        self.block_to_placements = self._build_block_mapping(self.placements)
        self._x = cp.Variable(num_placements, boolean=True)
        if warm_start is not None:
            self._x.value = warm_start
        
        constraints = self._build_constraints()
        objective = self._build_objective()

        print(f"    Model built ({time.monotonic()-t1:.2f}s)")
        print(f"Total setup: {time.monotonic()-t0:.2f}s")
        return cp.Problem(objective, constraints)

    def presolve(self, placements, costs):
        t0 = time.monotonic()

        pruned_placements, pruned_num_placements, pruned_costs = self._prune_placements(placements, costs)
        warm_start = self._warm_start(pruned_placements, pruned_costs)

        print(f"    Presolve: {pruned_num_placements} retained, {len(placements)-pruned_num_placements} pruned ({time.monotonic()-t0:.2f}s)")
        return pruned_placements, pruned_num_placements, pruned_costs, warm_start
    
    def solve(self) -> SolverResult:
        problem = self.prepare()

        assert self._x is not None
        tol = self.settings.tolerance

        print("Solving...")
        problem.solve(verbose=True, solver='HIGHS', warm_start=True,
            highs_options={
                "mip_rel_gap": tol,
                })
        
        if problem.status != 'optimal':
            return SolverResult(False, problem.status)
        else:
            return SolverResult(True, problem.status, self._x.value, self.placements)

    def _warm_start(self, placements, costs):
        return np.zeros(len(placements))

    def _build_block_mapping(self, placements):
        block_to_placements = defaultdict(list)
        for i, (tile, anchor) in enumerate(placements):
            for coord in tile.anchor_footprint(anchor):
                block_to_placements[coord].append(i)
        return block_to_placements

    def _prune_placements(self, placements, costs):
        pruned_placements = []
        pruned_costs = []
        pruned_num_placements = 0

        for (tile, anchor), cost in zip(placements, costs):
            mean_cost = cost / np.sum(tile.footprint_mask)
            if mean_cost > self.settings.presolve_threshold:
                continue
            pruned_placements.append((tile, anchor))
            pruned_costs.append(cost)
            pruned_num_placements += 1

        return pruned_placements, pruned_num_placements, np.array(pruned_costs)

    def _build_costs(self):
        t0 = time.monotonic()
        s, img = self.settings, self.image_data
        raw_placements, raw_num_placements = self.tile_set.generate_placements(
            img.num_cols,
            img.num_rows
        )
        print(f"    Placements generated: {raw_num_placements} ({time.monotonic()-t0:.2f}s)")
        
        t0 = time.monotonic()
        edge_weight, size_weight = s.edge_penalty_weight, s.size_bonus_weight
        edge_grid, rgb_grid = img.laplacian_grid, img.rgb_grid / 255.0     # normalizing rgb grid to range [0, 1]
        raw_costs = []

        for (tile, (i, j)) in raw_placements:
            footprint_mask = tile.footprint_mask
            normalized_tile = tile.colour / 255.0

            rgb_window = rgb_grid[i:i+tile.height, j:j+tile.width]
            window_errors = (normalized_tile - rgb_window) ** 2
            rgb_error = np.sum(window_errors * footprint_mask[:, :, np.newaxis])
            
            edge_window = edge_grid[i:i+tile.height, j:j+tile.width]
            max_edge = np.max(edge_window * footprint_mask)

            edge_pen = edge_weight * max_edge * (tile.scale - 1)**2
            size_bonus = -size_weight * (tile.scale - 1)**2
            tile_cost = rgb_error + edge_pen + size_bonus

            raw_costs.append(tile_cost)

        print(f"    Costs computed ({time.monotonic()-t0:.2f}s)")
        return raw_placements, raw_num_placements, np.array(raw_costs)

    def _build_objective(self) -> cp.Minimize:
        assert self.costs is not None
        return cp.Minimize(self.costs @ self._x)
    
    def _build_constraints(self) -> list[cp.Constraint]:
        assert self._x is not None
        assert self.block_to_placements is not None
        constraints = []
        for coord, placement_indices in self.block_to_placements.items():
            constraints.append(cp.sum(self._x[placement_indices]) == 1)
        
        return constraints