import numpy as np
import os
import random
import json
import uuid
from PIL import Image
from pathlib import Path
from core import Polyomino, OptimizationSettings, ImageSettings, TileConfig, ImageData, TilingOptimizer, TilingRenderer

###############################
# Config
###############################

NUM_ROWS = 100
NUM_COLS = 160
BLOCK_SIZE = 8
SCALES = [1, 2]
EDGE_WEIGHT = 0.35
SIZE_BONUS = 0.15
SOURCE_NAME = 'starry-night'
PRESOLVE = True
PRESOLVE_THRESHOLD = 0.25
OPT_TOLERANCE = 0.08
AUTOPICK_COLOURS = True
NUM_COLOURS = 14

random.seed(42)
np.random.seed(42)

TARGET_IMAGE_PATH = os.path.join(os.path.dirname(__file__), f'sources/{SOURCE_NAME}.jpg')

output_dir = Path("output") / f"edge{EDGE_WEIGHT}-size{SIZE_BONUS}"
filename = f"{SOURCE_NAME}-{NUM_COLS}x{NUM_ROWS}"

if PRESOLVE:
    filename += f"-pre"
if AUTOPICK_COLOURS:
    filename += f"-auto{NUM_COLOURS}"

OUTPUT_IMAGE_PATH = output_dir / f"{filename}.png"
OUTPUT_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

image = Image.open(TARGET_IMAGE_PATH)

image_settings = ImageSettings(NUM_ROWS, NUM_COLS, BLOCK_SIZE, AUTOPICK_COLOURS, NUM_COLOURS)
optimization_settings = OptimizationSettings(EDGE_WEIGHT, SIZE_BONUS, OPT_TOLERANCE, PRESOLVE, PRESOLVE_THRESHOLD)

image_data = ImageData(image, image_settings)
if AUTOPICK_COLOURS:
    image_data.prepare_palette()
else:
    PALETTE_PATH = os.path.join(os.path.dirname(__file__), f'colours/{SOURCE_NAME}-colours.json')
    with open(PALETTE_PATH, "r") as f:
        palette_data = json.load(f)
    colours = np.array(palette_data['colours'])
    image_data.prepare_palette(colours)

###############################
# Tile setup
###############################

POLYOMINOES = [
    # L-tromino
    Polyomino(np.array([
        [1, 1],
        [1, 0]
    ])),

    # I-tromino
    Polyomino(np.array([
        [1, 1, 1]
    ])),

    # Horizontal Domino
    Polyomino(np.array([
        [1, 1]
    ])),

    Polyomino(np.array([
        [1, 1, 1],
        [1, 0, 0]
    ])),

    Polyomino(np.array([
        [1, 1, 1],
        [0, 1, 0]
    ])),

    Polyomino(np.array([
        [0, 1, 1],
        [1, 1, 0]
    ])),

    Polyomino(np.array([
        [1]
    ])),
]

tile_config = TileConfig(POLYOMINOES, SCALES)

###############################
# Create and solve model
###############################

tiling_optimizer = TilingOptimizer(image_data, tile_config, optimization_settings)
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