import pyprojroot
import sys
root = pyprojroot.here()
sys.path.append(str(root))
import os
import pytorch_lightning as pl
import os
from typing import List
from dataclasses import dataclass
from pathlib import Path
import pyprojroot
import src
from src.preprocessing.preprocess_sat import (
    preprocess_sentinel2
)
from src.preprocessing.subtile_esd_hw02 import (
    Subtile,
    restitch
)
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt 
import cv2


root = pyprojroot.here()
all_dir = root / "data" / "processed" / "4x4" / "Total" / "subtiles"
output_dir = root / "data" / "restitched"
output_dir_np = root / "data" / "restitched" / "restitched_np_viirs"
satellite_type = "viirs"

# gamma 2.2 for other images
# gamma 0.5 for oversaturated images
for tile_id in range(1,61):
    stitched_stack, _ = restitch(all_dir, output_dir, satellite_type, "Tile"+str(tile_id), (0,4), (0,4))
    #preprocessed_sentinel2 = preprocess_sentinel2(stitched_sen2, clip_quantile=0.05, gamma=0.7)
    print(stitched_stack.shape)
    np.save(output_dir_np / Path("Tile"+str(tile_id)), stitched_stack)
    # for time_ax in range(0, 4):
    #     #image = preprocessed_sentinel2[time_ax]
    #     image = image[[3,2,1], :, :]
    #     image = np.transpose(image, (1, 2, 0))
    #     print(image.shape)
    #     resized_image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    #     curr_dir = Path(output_dir / "restitched_imgs" / f"{"Tile"+str(tile_id)}")
    #     if not os.path.exists(curr_dir) and not os.path.isdir(curr_dir): 
    #         curr_dir.mkdir(parents=True, exist_ok=True)
    #     plt.imsave(curr_dir / f"{time_ax}.jpg", resized_image, vmin=-0.5, vmax=3.5)
