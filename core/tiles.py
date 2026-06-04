from collections import defaultdict

class Polyomino:
    '''
    A Polyomino shape, equipped with the ability to rotate and rescale
    '''
    def __init__(self, name: str, footprint: list, scale: int = 1):
        self.name = name
        self.footprint = footprint
        self.height = max(r for r,c in footprint) + 1
        self.width  = max(c for r,c in footprint) + 1
        self.scale = scale

    def rotate(self) -> 'Polyomino':
        rotated = [(c, -r) for r,c in self.footprint]
        min_r = min(r for r,c in rotated)
        min_c = min(c for r,c in rotated)
        rotated = [(r-min_r, c-min_c) for r,c in rotated]
        return Polyomino(self.name, rotated)

    def rotations(self) -> list['Polyomino']:
        rots = []
        s = self
        for _ in range(4):
            if not any(set(s.footprint) == set(r.footprint) for r in rots):
                rots.append(s)
            s = s.rotate()
        return rots
    
    def scaled(self, scale: int) -> 'Polyomino':
        scaled_footprint = []
        for dr, dc in self.footprint:
            for u in range(scale):
                for v in range(scale):
                    i = dr*scale + u
                    j = dc*scale + v
                    scaled_footprint.append((i,j))
        scaled = Polyomino(f"{self.name}_x{scale}", scaled_footprint, scale)
        return scaled
    
class Tile:
    '''
    A Polyomino tile with a colour
    '''
    def __init__(self, polyomino: Polyomino, colour: tuple[int, int, int]):
        self._polyomino = polyomino
        self.colour = colour

    def anchor_footprint(self, anchor: tuple) -> list[tuple]:
        anchored_footprint = []
        for block in self.footprint:
            anchored_footprint.append((block[0] + anchor[0], block[1] + anchor[1]))
        return anchored_footprint

    def rotations(self) -> list['Tile']:
        rots = [Tile(rot, self.colour) for rot in self._polyomino.rotations()]
        return rots
    
    def scaled(self, scale: int) -> 'Tile':
        return Tile(self._polyomino.scaled(scale), self.colour)
    
    @property
    def height(self) -> int:
        return self._polyomino.height
    
    @property
    def width(self) -> int:
        return self._polyomino.width
    
    @property
    def footprint(self) -> list[tuple]:
        return self._polyomino.footprint
    
    @property
    def scale(self) -> int:
        return self._polyomino.scale
    


class TileSet:
    '''
    A set of Polyomino tiles used to tile a rectangular plane
    '''
    def __init__(self, base_tiles: list[Tile], scales: list[int]):
        self.tiles = []

        copy_scales = list(scales)
        if 1 not in scales:
            copy_scales.insert(0, 1)
        self.scales = copy_scales

        for tile in base_tiles:
            for scale in self.scales:
                for rotation in tile.rotations():
                    self.tiles.append(rotation.scaled(scale))

    def generate_placements(self, num_cols: int, num_rows: int) -> tuple[list, dict, int]:
        placements = []
        block_to_placements = defaultdict(list)
        p = 0

        placements = [
            (tile, (i, j))
            for tile in self.tiles
            for i in range(num_rows - tile.height + 1)
            for j in range(num_cols - tile.width + 1)
            ]
        
        for p, (tile, anchor) in enumerate(placements):
            for block in tile.anchor_footprint(anchor):
                block_to_placements[block].append(p)

        return placements, block_to_placements, len(placements)