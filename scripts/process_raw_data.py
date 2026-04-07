import pyprojroot
import sys
root = pyprojroot.here()
sys.path.append(str(root))
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser
import os
import numpy as np
from typing import List
from dataclasses import dataclass
import shutil
import os
from pathlib import Path
from src.preprocessing.subtile_esd_hw02 import (
    restitch
)
from src.preprocessing.file_utils_GAN import copy_missing_files_and_directories
from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation

@dataclass
class ESDConfig:
    """
    IMPORTANT: This class is used to define the configuration for the experiment
    Please make sure to use the correct types and default values for the parameters
    and that the path for processed_dir contain the tiles you would like 
    """
    processed_dir: str | os.PathLike = root / 'data/processed/4x4'
    processed_dir_total: str | os.PathLike = root / 'data/processed/4x4/Total'
    restitched_dir = root / "data" / "restitched"
    restitched_image_dir = root / "data" / "restitched" /"restitched_imgs"
    restitched_np_dir = root / "data" / "restitched" /"restitched_np"
    raw_dir: str | os.PathLike = root / 'data/raw/Train'
    selected_bands: None = None
    tile_size_gt: int = 4
    batch_size: int = 8
    seed: int = 12378921

def process_data(options: ESDConfig):
    esd_datamodule = ESDDataModule(options.processed_dir, options.raw_dir, options.selected_bands, options.tile_size_gt, options.batch_size, options.seed)
    
    try:
        esd_datamodule.prepare_data()
    except Exception as e:
        print(e)
    esd_datamodule.setup("fit")
    if not os.path.exists(options.processed_dir_total) and not os.path.isdir(options.processed_dir_total): 
        Path(options.processed_dir_total).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(options.processed_dir_total / Path("metadata")) and not os.path.isdir(options.processed_dir_total / Path("metadata")): 
        (Path(options.processed_dir_total)/ Path("metadata")).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(options.processed_dir_total/ Path("subtiles")) and not os.path.isdir(options.processed_dir_total/ Path("subtiles")): 
        (Path(options.processed_dir_total)/ Path("subtiles")).mkdir(parents=True, exist_ok=True)

    copy_missing_files_and_directories(options.processed_dir / Path("Train") / Path("metadata"), options.processed_dir_total / Path("metadata"))
    copy_missing_files_and_directories(options.processed_dir / Path("Train") / Path("subtiles"), options.processed_dir_total / Path("subtiles"))
    copy_missing_files_and_directories(options.processed_dir / Path("Val") / Path("metadata"), options.processed_dir_total / Path("metadata"))
    copy_missing_files_and_directories(options.processed_dir / Path("Val") / Path("subtiles"), options.processed_dir_total / Path("subtiles"))


def dataset_to_np(options: ESDConfig):
    for tile_id in range(1,61):
        stitched_stack, _ = restitch(options.processed_dir_total / "subtiles", "sentinel2", "Tile"+str(tile_id), (0,4), (0,4))
        np.save(options.restitched_np_dir / Path("Tile"+str(tile_id)), stitched_stack)

if __name__ == '__main__':
    config = ESDConfig()
    process_data(config)
    dataset_to_np(config)