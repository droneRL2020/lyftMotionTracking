import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import logging
from torch import nn, optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from typing import Dict
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from utils import Utils
from config.global_conf import Global

class Inferencer():
    def __init__(self, W_PATH=None):
        assert W_PATH != None, "check W_PATH option"
        
        os.environ["L5KIT_DATA_FOLDER"] = f"{Global.DIR_INPUT}/l5kit/dataset"
        self.cfg = Global.getConfig()
        self.test_cfg = self.cfg["test_data_loader"]
        dm = LocalDataManager(None)
        rasterizer = build_rasterizer(self.cfg, dm)
        test_zarr = ChunkedDataset(dm.require(self.test_cfg["key"])).open()
        test_mask = np.load(f"{Global.DIR_INPUT}/l5kit/dataset/scenes/mask.npz")["arr_0"]
        self.test_dataset = AgentDataset(self.cfg, test_zarr, rasterizer, agents_mask=test_mask)
        self.test_dataloader = DataLoader(self.test_dataset,
                             shuffle=self.test_cfg["shuffle"],
                             batch_size=self.test_cfg["batch_size"],
                             num_workers=self.test_cfg["num_workers"])
        self.W_PATH = W_PATH
        