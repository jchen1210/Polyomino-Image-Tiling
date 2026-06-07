from skimage import filters
from PIL import Image
from .config import ImageSettings
import numpy as np

class ImageData:
    '''
    Computes and houses image data necessary for creating a tiling of the image
    '''
    def __init__(self, img, settings: ImageSettings):
        self.settings = settings

        num_rows = settings.num_rows
        num_cols = settings.num_cols
        block_size = settings.block_size
        img = img.resize((num_cols*block_size, num_rows*block_size), Image.LANCZOS) # type: ignore

        greyscale_img = img.convert("L")
        self._greyscale_px = np.array(greyscale_img)
        self._img_px = np.array(img)
        self._laplacian_grid = None
        self._brightness_grid = None
        self._rgb_grid = None

    @property
    def laplacian_grid(self):
        if self._laplacian_grid is None:
            self._laplacian_grid = self._compute_laplacian_grid()
        return self._laplacian_grid
        
    @property
    def rgb_grid(self):
        if self._rgb_grid is None:
            self._rgb_grid = self._compute_rgb_grid() / 255.0
        return self._rgb_grid
    
    @property
    def num_rows(self):
        return self.settings.num_rows
    
    @property
    def num_cols(self):
        return self.settings.num_cols
    
    def _compute_laplacian_grid(self):
        num_rows = self.num_rows
        num_cols = self.num_cols
        block_size = self.settings.block_size

        laplace_px = filters.laplace(self._greyscale_px / 255.0)

        laplacian_grid = np.zeros((num_rows, num_cols))
        for i in range(num_rows):
            for j in range(num_cols):
                block = laplace_px[i*block_size:(i+1)*block_size,
                                   j*block_size:(j+1)*block_size]
                laplacian_grid[i,j] = np.mean(np.abs(block))

        laplacian_grid /= np.max(laplacian_grid)
        laplacian_grid = np.clip(laplacian_grid, 0, 1)

        return laplacian_grid
    
    def _compute_rgb_grid(self):
        num_rows = self.num_rows
        num_cols = self.num_cols
        block_size = self.settings.block_size
        img_px = self._img_px

        rgb_grid = np.zeros((num_rows, num_cols, 3))
        for i in range(num_rows):
            for j in range(num_cols):
                block = img_px[i*block_size:(i+1)*block_size,
                               j*block_size:(j+1)*block_size]
                rgb_grid[i, j] = np.mean(block, axis=(0, 1))

        return rgb_grid


class Palette:
    '''
    A colour palette used for tiling the image
    '''
    def __init__(self, colours: list[tuple]):
        self.colours = colours
        self.num_colours = len(colours)