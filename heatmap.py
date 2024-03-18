from typing import Tuple
import numpy as np

def gen_binary_map(wh: Tuple[int, int],
                   cxy: Tuple[float, float],
                   r: float,
                   data_type: np.dtype = np.float32,
):
    w, h   = wh
    cx, cy = cxy
    if cx < 0 or cy < 0:
        return np.zeros((h,w), dtype=data_type)
    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    distmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
    bmap    = np.zeros_like(distmap)
    bmap[distmap <= r**2] = 1
    return bmap.astype(data_type)