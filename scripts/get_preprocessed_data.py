import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import pyprojroot
root = pyprojroot.here()
sys.path.append(str(root))
from src.preprocessing.subtile_esd_hw02 import grid_slice
from src.preprocessing.file_utils import load_satellite
from src.preprocessing.subtile_esd_hw02 import grid_slice
import argparse
root = pyprojroot.here()


def npy_to_train_validate(train_metadata_dir, val_metadata_dir, restitched_gan_np_dir, output_dir):
    val_set = set()

    train_dir = (Path(output_dir) / Path("Train"))
    val_dir = (Path(output_dir) / Path("Val"))
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for tile in Path(val_metadata_dir).iterdir():
        val_set.add(tile.name.split("_")[0])

    for np_file in restitched_gan_np_dir.iterdir():
        tile_name = np_file.name.split('.')[0]
        sen2_stack = np.load(np_file)
        tile_raw_dir = root / "data" / "raw" / "Train" / tile_name

        satellite_stack = {}
        satellite_metadata = {}

        _, metadata = load_satellite(tile_raw_dir, "sentinel2")
        gt_stack, gt_metadata = load_satellite(tile_raw_dir, "gt")
        satellite_stack["sentinel2"] = sen2_stack
        satellite_metadata["sentinel2"] = metadata
        satellite_stack["gt"] = gt_stack
        satellite_metadata["gt"] = gt_metadata

        subtiles = grid_slice(satellite_stack, satellite_metadata, 4)

        if tile_name in val_set:
            for index in range(0, len(subtiles)):
                subtiles[index].save(val_dir)
        else:
            for index in range(0, len(subtiles)):
                subtiles[index].save(train_dir)

if __name__ == '__main__':
    root_mask_dir = root / "data" / "masks"
    restitched_image_dir = root / "data" / "restitched" /"restitched_imgs"
    gan_imgs_dir = root / "data" / "restitched" /"gan_imgs"
    restitched_np_dir = root / "data" / "restitched" /"restitched_np"
    restitched_gan_np_dir = root / "data" / "restitched" /"restitched_gan_np"
    train_metadata_dir = root / "data" / "processed"/ "4x4" /"Train" / "metadata"
    val_metadata_dir = root / "data" / "processed"/ "4x4" /"Val" / "metadata"
    preprocessed_dir = root / "data" / "preprocessed"
    npy_to_train_validate(train_metadata_dir, val_metadata_dir, restitched_gan_np_dir, preprocessed_dir)