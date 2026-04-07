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
import pyprojroot
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

root = pyprojroot.here()
processed_dir = root / "data" / "processed" / "4x4"
metadata_dict = {}

for metadata in Path(processed_dir / "Val" / "metadata").iterdir():
    with open(metadata) as file:
        data = json.load(file)
        metadata_dict[metadata.name.split('.')[0] + ".npz"] = data

# print(metadata_dict)
for k,v in metadata_dict.items():
    print(f': {v}')
    break

for subtile in Path(processed_dir/ "Val" / "subtiles"):
    continue
    restitch_eval(subtile, "viirs", )
    pass
