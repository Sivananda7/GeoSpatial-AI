import pyprojroot
import sys
import os
root = pyprojroot.here()
sys.path.append(str(root))
import pytorch_lightning as pl
from argparse import ArgumentParser
import os
from typing import List
from dataclasses import dataclass
from pathlib import Path

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary
)

from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.preprocessing.subtile_esd_hw02 import Subtile
from src.visualization.restitch_plot import (
    restitch_eval,
    restitch_and_plot
)
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
import tifffile
import json
@dataclass
class EvalConfig:
    processed_dir: str | os.PathLike = root / 'data/processed/4x4'
    raw_dir: str | os.PathLike = root / 'data/raw/Train'
    results_dir: str | os.PathLike = root / 'data/predictions'/ "UNet"
    selected_bands: None = None
    tile_size_gt: int = 4
    batch_size: int = 8
    seed: int = 12378921
    num_workers: int = 50
    model_path: str | os.PathLike = root / "models" / "UNet" / "last.ckpt"
    
def main(options):
    """
    Prepares datamodule and loads model, then runs the evaluation loop

    Inputs:
        options: EvalConfig
            options for the experiment
    """
    # Load datamodule
    datamodule = ESDDataModule(options.processed_dir, options.raw_dir, options.selected_bands)
    datamodule.setup("fit")
    model = ESDSegmentation.load_from_checkpoint(options.model_path)
    # model = datamodule.load_from_checkpoint(options.model_path)


    # set the model to evaluation mode (model.eval())
    # this is important because if you don't do this, some layers
    # will not evaluate properly
    model.to('cuda')
    model.eval()

    # instantiate pytorch lightning trainer
    trainer = pl.Trainer(logger=False)

    # run the validation loop with trainer.validate
    trainer.validate(model, dataloaders=datamodule)

    # run restitch_and_plot


    tiles = []

    metadata_dict = {}

    for metadata in Path(options.processed_dir / "Val" / "metadata").iterdir():
        with open(metadata) as file:
            data = json.load(file)
            metadata_dict[metadata.name.split('.')[0] + ".npz"] = data
    visitedTiles = set()
    for subtile in Path(options.processed_dir/ "Val" / "subtiles").iterdir():
        data = metadata_dict[subtile.name]
        parent_tile_id = data["parent_tile_id"]

        if parent_tile_id in visitedTiles:
            continue
        else:
            visitedTiles.add(parent_tile_id)
        image, gt, prediction = restitch_eval(options.processed_dir, "sentinel2", data["parent_tile_id"], (0,data["subtile_size"]), (0,data["subtile_size"]), datamodule, model)
        #print(prediction.squeeze(0))
        best_class = np.argmax(prediction.squeeze(0), axis=0)
        #print(best_class)

        # restitch and plot for testing
        restitch_and_plot(parent_tile_id, data["x_gt"], data["y_gt"], image, gt, best_class, "sentinel2", [3,2,1], root / "data" / "test_restitch_plots")
        if not os.path.exists(options.results_dir) and not os.path.isdir(options.results_dir): 
            results_dir = options.results_dir
            if type(options.results_dir) is not str:
                results_dir = Path(options.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)


        tiles.append((parent_tile_id, best_class))
    
    for parent_tile_id, y_pred in tiles:

        # freebie: plots the predicted image as a jpeg with the correct colors
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Settlements", np.array(['#ff0000', '#0000ff', '#ffff00', '#b266ff']), N=4)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(y_pred, vmin=-0.5, vmax=3.5,cmap=cmap)
        plt.savefig(options.results_dir / f"{parent_tile_id}.png")
    

if __name__ == '__main__':
    config = EvalConfig()
    parser = ArgumentParser()

    parser.add_argument("--model_path", type=str, help="Model path.", default=config.model_path)
    parser.add_argument("--raw_dir", type=str, default=config.raw_dir, help='Path to raw directory')
    parser.add_argument("-p", "--processed_dir", type=str, default=config.processed_dir,
                        help=".")
    parser.add_argument("--results_dir", type=str, default=config.results_dir, help="Results dir")
    main(EvalConfig(**parser.parse_args().__dict__))