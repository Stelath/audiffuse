import numpy as np
import pandas as pd
import argparse

import torch
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI

from audiffuse import Audiffuse
from es_dataset import EpidemicSoundDataset, EpidemicSoundDataModule


def cli_main():
    cli = LightningCLI(Audiffuse, EpidemicSoundDataModule, seed_everything_default=42)


if __name__ == "__main__":
    cli_main()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-d', '--dataset', type=str)
#     parser.add_argument('-h', '--hyperparameters', type=str)
#     args = parser.parse_args()

#     main()