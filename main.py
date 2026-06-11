import numpy as np
import os
import random
import json
import uuid
from PIL import Image
from pathlib import Path
from core import Polyomino, TileSet, Tile, OptimizationSettings, ImageSettings, ImageData, Palette, TilingOptimizer, TilingRenderer

###############################
# Problem Dimensions
###############################

NUM_ROWS = 160
NUM_COLS = 256
BLOCK_SIZE = 8
SCALES = [1, 2, 4, 8]
EDGE_WEIGHT = 0.35
SIZE_BONUS = 0.15
SOURCE_NAME = 'starry-night'
PRESOLVE = True
PRESOLVE_THRESHOLD = 0.25
OPT_TOLERANCE = 0.05

random.seed(42)
np.random.seed(42)

###############################
# Config
###############################

TARGET_IMAGE_PATH = os.path.join(os.path.dirname(__file__), f'sources/{SOURCE_NAME}.jpg')

output_dir = Path("output") / f"edge{EDGE_WEIGHT}-size{SIZE_BONUS}"
filename = f"{SOURCE_NAME}-{NUM_COLS}x{NUM_ROWS}"

if PRESOLVE:
    filename += f"-pre"

OUTPUT_IMAGE_PATH = output_dir / f"{filename}.png"
OUTPUT_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

image = Image.open(TARGET_IMAGE_PATH)

PALETTE_PATH = os.path.join(os.path.dirname(__file__), f'colours/{SOURCE_NAME}-colours.json')
with open(PALETTE_PATH, "r") as f:
    palette_data = json.load(f)
colours = [tuple(colour) for colour in palette_data['colours']]

image_settings = ImageSettings(NUM_ROWS, NUM_COLS, BLOCK_SIZE)
optimization_settings = OptimizationSettings(EDGE_WEIGHT, SIZE_BONUS, OPT_TOLERANCE, PRESOLVE, PRESOLVE_THRESHOLD)
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
    assert result.placements is not None
    
    print('Problem status: optimal')
    renderer = TilingRenderer(result.values, result.placements, image_settings)
    output_image = renderer.render()

    output_image.save(OUTPUT_IMAGE_PATH)
    print("Saved:", OUTPUT_IMAGE_PATH)
else:
    print(f'Problem status: {result.status}')
    print(f'No output image was generated')