import random
import time
from pathlib import Path
from typing import List

from einops import rearrange

import imageio.v3 as iio
import numpy as np
from numpy import load

import tyro
from tqdm.auto import tqdm

import viser

def main() -> None:
    """Visualize COLMAP sparse reconstruction outputs.

    Args:
        colmap_path: Path to the COLMAP reconstruction directory.
        images_path: Path to the COLMAP images directory.
        downsample_factor: Downsample factor for the images.
    """
    server = viser.ViserServer()
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    data = load('./test.npy', allow_pickle=True)
    dictionary = data.item()  # Extracts the single item from a 0D array
    pts3d = dictionary['pts3d']
    points3d = rearrange(pts3d[0], "h w c -> (h w) c")
    colors = rearrange(dictionary['img'][0][0], 'c h w -> (h w) c')

    points = np.array(points3d)
    colors = colors.cpu().numpy()

    # points = np.array([points3d[p_id] for p_id in points3d])
    # colors = np.array([points3d[p_id] for p_id in points3d])
    
    point_cloud = server.scene.add_point_cloud(
        name="/pcd",
        points=points,
        colors=colors,
        point_size=0.01,
    )   

    while True:
        time.sleep(1e-3)


if __name__ == "__main__":
    tyro.cli(main)
