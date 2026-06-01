import cvxpy as cp
import numpy as np
from PIL import Image, ImageDraw
from skimage import filters
from collections import defaultdict
import os
import random
import uuid
import json
from core import Polyomino, TileSet, Tile

###############################
# Problem Dimensions
###############################

NUM_ROWS = 20
NUM_COLS = 30
BLOCK_SIZE = 8
SCALES = [1, 2]
EDGE_WEIGHT = 5.0
SIZE_BONUS = 2.0

random.seed(42)
np.random.seed(42)

###############################
# Polyomino definitions
###############################

POLYOMINOES = [
    Polyomino("L",  [(0,0),(0,1),(1,0)]),
    Polyomino("I3", [(0,0),(0,1),(0,2)]),
    Polyomino("D2h",[(0,0),(0,1)]),
    Polyomino("D2v",[(0,0),(1,0)])
]

###############################
# Load palette config
###############################

PALETTE_CONFIG = os.path.join(os.path.dirname(__file__), "colors/starry-night.json")

with open(PALETTE_CONFIG, "r") as f:
    palette_data = json.load(f)

palette = [tuple(color) for color in palette_data["colors"]]
NUM_COLORS = len(palette)

###############################
# Load target image 
###############################

TARGET_IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'sources/starry-night.jpg')
OUTPUT_IMAGE = f"output/edge-aware-v1-monalisa-{uuid.uuid4().hex}.png"

img = Image.open(TARGET_IMAGE_PATH).convert("L")
img = img.resize((NUM_COLS*BLOCK_SIZE, NUM_ROWS*BLOCK_SIZE), Image.LANCZOS)
img_arr = np.array(img)

block_brightness = np.zeros((NUM_ROWS, NUM_COLS))
for i in range(NUM_ROWS):
    for j in range(NUM_COLS):
        block = img_arr[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE,
                        j*BLOCK_SIZE:(j+1)*BLOCK_SIZE]
        block_brightness[i,j] = round(block.mean() / 255.0 * (NUM_COLORS - 1))

###############################
# Edge map
###############################

laplace_edges = filters.laplace(img_arr / 255.0)

edge_block = np.zeros((NUM_ROWS, NUM_COLS))
for i in range(NUM_ROWS):
    for j in range(NUM_COLS):
        block = laplace_edges[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE,
                              j*BLOCK_SIZE:(j+1)*BLOCK_SIZE]
        edge_block[i,j] = np.mean(np.abs(block))

edge_block /= np.max(edge_block)
edge_block = np.clip(edge_block, 0, 1)

###############################
# Palette
###############################

def generate_palette():
    tiles = []
    brightness_vals = []

    for (r,g,b) in palette:
        tile = Image.new("RGB", (BLOCK_SIZE, BLOCK_SIZE), (r,g,b))
        tiles.append(tile)

        brightness = 0.299*r + 0.587*g + 0.114*b
        brightness_vals.append(brightness)

    brightness_vals = np.array(brightness_vals)
    return tiles, brightness_vals

colored_tiles, brightness_values = generate_palette()

normalized_brightness = (brightness_values - brightness_values.min()) / \
                        (brightness_values.max() - brightness_values.min()) * 9

color_to_brightness = {(r, g, b) : normalized_brightness[i] for i, (r, g, b) in enumerate(palette)}
colour_to_index = {(r, g, b) : i for i, (r, g, b) in enumerate(palette)}

###############################
# Placement generation
###############################

tiles = [Tile(POLYOMINOES[i % len(POLYOMINOES)], tuple(color)) for i, color in enumerate(palette_data["colors"])]
tileset = TileSet(tiles, SCALES)

tileset.set_placements(NUM_COLS, NUM_ROWS)
placements = tileset.placements
block_to_placements = tileset.block_to_placements()

NUM_PLACEMENTS = len(placements)
print("Total placements:", NUM_PLACEMENTS)

###############################
# CVXPY model
###############################

x = cp.Variable(NUM_PLACEMENTS, boolean=True)

constraints = []
for i in range(NUM_ROWS):
    for j in range(NUM_COLS):
        constraints.append(cp.sum(x[block_to_placements[(i,j)]]) == 1)

###############################
# Objective
###############################

costs = np.zeros(NUM_PLACEMENTS)

for p, (tile, (i,j)) in enumerate(placements):
    err = 0
    max_edge = 0
    for (ii,jj) in tile.anchor_footprint((i, j)):
        err += (normalized_brightness[colour_to_index[tile.colour]] - block_brightness[ii,jj])**2
        max_edge = max(max_edge, edge_block[ii,jj])

    edge_pen = EDGE_WEIGHT * max_edge * (tile.scale - 1)**2
    size_bonus = -SIZE_BONUS * (tile.scale - 1)

    costs[p] = err + edge_pen + size_bonus



problem = cp.Problem(cp.Minimize(costs @ x), constraints)

###############################
# Solve
###############################

print("Solving...")
problem.solve(verbose=True, solver='HIGHS',
              highs_options={
                "mip_rel_gap": 0,
                })
print("Status:", problem.status)

###############################
# Render image
###############################

result = Image.new("RGB",(NUM_COLS*BLOCK_SIZE,NUM_ROWS*BLOCK_SIZE),(255,255,255))
draw = ImageDraw.Draw(result)

for p, val in enumerate(x.value):
    if val > 0.5:
        tile, (i, j) = placements[p]
        footprint = tile.anchor_footprint((i, j))
        footprint_set = set(footprint)

        for (ii,jj) in footprint:
            result.paste(
                colored_tiles[colour_to_index[tile.colour]],
                (jj*BLOCK_SIZE, ii*BLOCK_SIZE)
            )

        # borders
        for (ii,jj) in footprint:
            x0 = jj * BLOCK_SIZE
            y0 = ii * BLOCK_SIZE
            x1 = x0 + BLOCK_SIZE
            y1 = y0 + BLOCK_SIZE

            if (ii-1, jj) not in footprint_set:
                draw.line([(x0,y0),(x1,y0)], fill=(0,0,0), width=2)
            if (ii+1, jj) not in footprint_set:
                draw.line([(x0,y1),(x1,y1)], fill=(0,0,0), width=2)
            if (ii, jj-1) not in footprint_set:
                draw.line([(x0,y0),(x0,y1)], fill=(0,0,0), width=2)
            if (ii, jj+1) not in footprint_set:
                draw.line([(x1,y0),(x1,y1)], fill=(0,0,0), width=2)

result.save(OUTPUT_IMAGE)
print("Saved:", OUTPUT_IMAGE)
