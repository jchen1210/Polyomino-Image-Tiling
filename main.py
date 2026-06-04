import numpy as np
from PIL import Image
import os
import random
import json
from core import Polyomino, TileSet, Tile, OptimizationSettings, ImageSettings, ImageData, Palette, TilingOptimizer, TilingRenderer

###############################
# Problem Dimensions
###############################

NUM_ROWS = 20
NUM_COLS = 30
BLOCK_SIZE = 8
SCALES = [1, 2]
EDGE_WEIGHT = 4.0
SIZE_BONUS = 3.0

random.seed(42)
np.random.seed(42)

###############################
# Config
###############################

TARGET_IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'sources/starry-night.jpg')
OUTPUT_IMAGE_PATH = f"output/new-starrynight.png"
image = Image.open(TARGET_IMAGE_PATH)

PALETTE_PATH = os.path.join(os.path.dirname(__file__), 'colours/starry-night-colours.json')
with open(PALETTE_PATH, "r") as f:
    palette_data = json.load(f)
colours = [tuple(colour) for colour in palette_data['colours']]

image_settings = ImageSettings(NUM_ROWS, NUM_COLS, BLOCK_SIZE)
optimization_settings = OptimizationSettings(EDGE_WEIGHT, SIZE_BONUS)
palette = Palette(colours)
image_data = ImageData(image, image_settings)

###############################
# Tile setup
###############################

POLYOMINOES = [
    Polyomino("L",  [(0,0),(0,1),(1,0)]),
    Polyomino("I3", [(0,0),(0,1),(0,2)]),
    Polyomino("D2h",[(0,0),(0,1)]),
    Polyomino("D2v",[(0,0),(1,0)])
]

tiles = [Tile(POLYOMINOES[i % len(POLYOMINOES)], tuple(colour)) for i, colour in enumerate(palette.colours)]
tileset = TileSet(tiles, SCALES)
tileset.set_placements(NUM_COLS, NUM_ROWS)

###############################
# Create and solve model
###############################

tiling_optimizer = TilingOptimizer(image_data, tileset, optimization_settings, palette)
result = tiling_optimizer.solve()

###############################
# Render image
###############################

if result.is_success:
    assert result.values is not None
    
    print('Problem status: optimal')
    renderer = TilingRenderer(tileset, palette, result.values, image_settings)
    output_image = renderer.render()

    output_image.save(OUTPUT_IMAGE_PATH)
    print("Saved:", OUTPUT_IMAGE_PATH)
else:
    print(f'Problem status: {result.status}')
    print(f'No output image was generated')