import numpy as np

class Polyomino:
    '''
    A Polyomino shape, equipped with the ability to rotate and rescale
    '''
    def __init__(self, footprint: np.ndarray, scale: int = 1):
        self.footprint_mask = footprint
        self._footprint_list = None
        self.height, self.width = np.shape(footprint)
        self.scale = scale

    def rotate(self) -> 'Polyomino':
        rotated_mask = np.rot90(self.footprint_mask).copy()
        return Polyomino(rotated_mask, self.scale)

    def rotations(self) -> list['Polyomino']:
        rotations = []
        seen_shapes = set()
        for k in range(4):
            rotated_mask = np.rot90(self.footprint_mask, k).copy()
            shape_sig = rotated_mask.tobytes()

            if shape_sig not in seen_shapes:
                seen_shapes.add(shape_sig)
                rotations.append(Polyomino(rotated_mask, self.scale))
            else:
                break
        return rotations

    def scaled(self, scale: int) -> 'Polyomino':
        scaling_matrix = np.ones((scale, scale))
        scaled_mask = np.kron(self.footprint_mask, scaling_matrix)
        return Polyomino(scaled_mask, scale)
    
    @property
    def footprint_list(self):
        if self._footprint_list is None:
            self._footprint_list = [tuple(coord) for coord in np.argwhere(self.footprint_mask == 1)]
        return self._footprint_list
    
class Tile:
    '''
    A Polyomino tile with a colour
    '''
    def __init__(self, polyomino: Polyomino, colour: tuple[int, int, int]):
        self._polyomino = polyomino
        self.colour = colour

    def anchor_footprint(self, anchor: tuple) -> list[tuple]:
        anchored_footprint = []
        for block in self.footprint_list:
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
    def footprint_list(self) -> list[tuple]:
        return self._polyomino.footprint_list
    
    @property
    def footprint_mask(self) -> np.ndarray:
        return self._polyomino.footprint_mask
    
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

    def generate_placements(self, num_cols: int, num_rows: int) -> tuple[list, int]:
        placements = [
            (tile, (i, j))
            for tile in self.tiles
            for i in range(num_rows - tile.height + 1)
            for j in range(num_cols - tile.width + 1)
            ]

        return placements, len(placements)