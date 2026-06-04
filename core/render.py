import numpy as np
from PIL import Image, ImageDraw
from .image_processor import Palette
from .config import ImageSettings
from .tiles import TileSet


class TilingRenderer:
    '''
    Renders an image tiling solution
    '''
    def __init__(self, solution: np.ndarray, placements: list[tuple], image_settings: ImageSettings):
        self.solution = solution
        self.settings = image_settings
        self.placements = placements

    def render(self):
        num_rows = self.settings.num_rows
        num_cols = self.settings.num_cols
        block_size = self.settings.block_size
        placements = self.placements

        result = Image.new("RGB",(num_cols*block_size,num_rows*block_size),(255,255,255))
        draw = ImageDraw.Draw(result)

        for p, val in enumerate(self.solution):
            if val > 0.5:
                tile, (i, j) = placements[p]
                footprint = tile.anchor_footprint((i, j))
                footprint_set = set(footprint)

                for (ii,jj) in footprint:
                    result.paste(
                        Image.new("RGB", (block_size, block_size), (tile.colour)),
                        (jj*block_size, ii*block_size)
                    )

                # borders
                for (ii,jj) in footprint:
                    x0 = jj * block_size
                    y0 = ii * block_size
                    x1 = x0 + block_size
                    y1 = y0 + block_size

                    if (ii-1, jj) not in footprint_set:
                        draw.line([(x0,y0),(x1,y0)], fill=(0,0,0), width=2)
                    if (ii+1, jj) not in footprint_set:
                        draw.line([(x0,y1),(x1,y1)], fill=(0,0,0), width=2)
                    if (ii, jj-1) not in footprint_set:
                        draw.line([(x0,y0),(x0,y1)], fill=(0,0,0), width=2)
                    if (ii, jj+1) not in footprint_set:
                        draw.line([(x1,y0),(x1,y1)], fill=(0,0,0), width=2)

        return result