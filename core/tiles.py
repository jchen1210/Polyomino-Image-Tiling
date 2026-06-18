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
            shape_sig = tuple(map(tuple, rotated_mask))

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
    def __init__(self, polyomino: Polyomino, colour: np.ndarray):
        if  colour.shape != (3,):
            raise ValueError(
                f"Palette must be a 1D array of shape (3,). "
                f"Got an array with shape {colour.shape} instead."
            )
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
    def __init__(self, base_shapes: list[Polyomino], scales: list[int], palette: np.ndarray):
        self.tiles = []
        base_tiles = self.assign_colours(base_shapes, palette)

        copy_scales = list(scales)
        if 1 not in scales:
            copy_scales.insert(0, 1)
        self.scales = copy_scales

        for tile in base_tiles:
            for scale in self.scales:
                for rotation in tile.rotations():
                    self.tiles.append(rotation.scaled(scale))

    def assign_colours(self, shapes: list[Polyomino], palette: np.ndarray) -> list[Tile]:
        '''
        Rotates through the polyomino shapes and assigns colours until no more colours are left in the palette.
        Shapes may end up with different numbers of assigned colours
        '''
        tiles = [Tile(shapes[i % len(shapes)], colour) for i, colour in enumerate(palette)]
        return tiles

    def generate_placements(self, num_cols, num_rows) -> tuple[list, int]:
        placements = [
            (tile, (i, j))
            for tile in self.tiles
            for i in range(num_rows - tile.height + 1)
            for j in range(num_cols - tile.width + 1)
            ]

        return placements, len(placements)