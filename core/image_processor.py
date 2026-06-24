import warnings
import numpy as np
from skimage import filters
from PIL import Image
from .config import ImageSettings
from sklearn.cluster import KMeans
from typing import Optional

class ImageData:
    '''
    Computes and houses image data necessary for creating a tiling of the image
    '''
    DEFAULT_COLOUR_BUDGET: int = 8

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

        self._palette = None
    
    def prepare_palette(self, palette: Optional[np.ndarray] = None):
        s = self.settings
        if s.auto_pick:
            if palette is not None:
                warnings.warn(
                    "Autopick was selected, but custom palette was provided\n"
                    "Autopicking colours anyways...",
                    RuntimeWarning
                )
            if s.requested_colour_budget is None:
                warnings.warn(
                    "Autopick was selected, but the number of colours was not configured\n"
                    "Using default colour budget of 8...",
                    RuntimeWarning
                )
            colour_budget = s.requested_colour_budget or self.DEFAULT_COLOUR_BUDGET
            self._auto_pick_palette(colour_budget)
        else:
            if palette is None:
                raise ValueError(
                    "Autopick was not selected but no custom palette was provided"
                )
            else:
                self._use_palette(palette)

    @property
    def palette(self):
        if self._palette is None:
            raise RuntimeError("Palette has not been initialized")
        return self._palette

    @property
    def num_colours(self):
        if self._palette is None:
            raise RuntimeError("Palette has not been initialized")
        return len(self._palette)

    @property
    def laplacian_grid(self):
        if self._laplacian_grid is None:
            self._laplacian_grid = self._compute_laplacian_grid()
        return self._laplacian_grid
        
    @property
    def rgb_grid(self):
        if self._rgb_grid is None:
            self._rgb_grid = self._compute_rgb_grid()
        return self._rgb_grid
    
    @property
    def num_rows(self):
        return self.settings.num_rows
    
    @property
    def num_cols(self):
        return self.settings.num_cols
    
    def _use_palette(self, palette: np.ndarray):
        if palette.ndim != 2 or palette.shape[1] != 3:
            raise ValueError(
                f"Palette must be a 2D array of shape (N, 3). "
                f"Got an array with shape {palette.shape} instead."
            )
        self._palette = palette

    def _auto_pick_palette(self, num_colours):
        X = self._img_px.reshape(-1, self._img_px.shape[-1])
        saturation = (np.max(X, axis=1) - np.min(X, axis=1)) / np.max(X, axis=1)
        saturation[np.isnan(saturation)] = 0
        kmeans = KMeans(n_clusters=num_colours, init='k-means++', random_state=42)
        kmeans.fit(X, sample_weight=saturation**2)
        centroids = kmeans.cluster_centers_
        self._palette = np.clip(np.round(centroids), 0, 255).astype(int)
        return self._palette
    
    def _compute_laplacian_grid(self):
        num_rows = self.num_rows
        num_cols = self.num_cols
        block_size = self.settings.block_size

        laplace_px = filters.laplace(self._greyscale_px)

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