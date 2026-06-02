from skimage import filters
from PIL import Image, ImageDraw
import os
from .config import ImageSettings
import numpy as np
import json

NUM_COLOURS = 10

class ImageData:
    '''
    Computes and houses image data necessary for creating a tiling of the image
    '''
    def __init__(self, image_path: str, settings: ImageSettings):
        self.settings = settings

        num_rows = settings.num_rows
        num_cols = settings.num_cols
        block_size = settings.block_size

        img = Image.open(image_path).convert("L")
        img = img.resize((num_cols*block_size, num_rows*block_size), Image.LANCZOS) # type: ignore
        self.img_arr = np.array(img)

    @property
    def laplacian(self):
        num_rows = self.settings.num_rows
        num_cols = self.settings.num_cols
        block_size = self.settings.block_size

        laplace_edges = filters.laplace(self.img_arr / 255.0)

        edge_block = np.zeros((num_rows, num_cols))
        for i in range(num_rows):
            for j in range(num_cols):
                block = laplace_edges[i*block_size:(i+1)*block_size,
                                      j*block_size:(j+1)*block_size]
                edge_block[i,j] = np.mean(np.abs(block))

        edge_block /= np.max(edge_block)
        edge_block = np.clip(edge_block, 0, 1)

        self.edge_arr = edge_block
        return self.edge_arr

    @property
    def brightness_arr(self):
        num_rows = self.settings.num_rows
        num_cols = self.settings.num_cols
        block_size = self.settings.block_size

        block_brightness = np.zeros((num_rows, num_cols))
        for i in range(num_rows):
            for j in range(num_cols):
                block = self.img_arr[i*block_size:(i+1)*block_size,
                                     j*block_size:(j+1)*block_size]
                block_brightness[i,j] = round(block.mean() / 255.0 * (NUM_COLOURS - 1))
        
        return block_brightness


class Palette:
    '''
    A colour palette used for tiling the image
    '''
    def __init__(self, path: str):
        with open(path, "r") as f:
            palette_data = json.load(f)

        self.colours = [tuple(colour) for colour in palette_data["colours"]]
        self.num_colours = len(self.colours)

    def colour_to_brightness(self):
        brightness_values = []
        for (r, g, b) in self.colours:
            brightness = 0.299*r + 0.587*g + 0.114*b
            brightness_values.append(brightness)

        brightness_values = np.array(brightness_values)
        normalized_brightness = (brightness_values - brightness_values.min()) / \
                            (brightness_values.max() - brightness_values.min()) * 9
        self.colour_to_brightness_dict = dict(zip(self.colours, normalized_brightness))
        return self.colour_to_brightness_dict