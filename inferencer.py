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
from l5kit.evaluation import write_pred_csv

from utils import Utils
from config.global_conf import Global
from net import TopkNet

class Inferencer():
    def __init__(self, W_PATH=None):
        print("W_PATH is", W_PATH)
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
    
    def run(self):
        model = TopkNet(self.cfg).to(Global.DEVICE)
        Global.load_weight(model, self.W_PATH)
        model.eval() 

        future_coords_offsets_pd = []
        timestamps = []
        agent_ids = []
        confs = []

        with torch.no_grad():
            dataiter = tqdm(self.test_dataloader)
    
            for data in dataiter:
                inputs = data["image"].to(Global.DEVICE)
                images = []
                samples_means, means, mixture_weights = model(inputs)

                fit_outputs = torch.stack([mean for mean in means], dim=1)
                
                fit_confidences = torch.stack([mixture_weight for mixture_weight in mixture_weights], dim=1).squeeze()
                outputs = torch.zeros(fit_outputs.size(0), fit_outputs.size(1), fit_outputs.size(2), 2).to(Global.DEVICE)

                conf = []
                one_hot_el = torch.eye(3,3)
                for i in range(fit_confidences.size(0)):
                    idx = torch.argmax(fit_confidences[i]).item()
                    conf.append(one_hot_el[idx])
            
                outputs[:,0] = Utils.map_writer_from_image_to_world(data, fit_outputs[:,0,:,:], self.cfg)
                outputs[:,1] = Utils.map_writer_from_image_to_world(data, fit_outputs[:,1,:,:], self.cfg)
                outputs[:,2] = Utils.map_writer_from_image_to_world(data, fit_outputs[:,2,:,:], self.cfg)
                
                future_coords_offsets_pd.append(outputs.cpu().numpy().copy())
                timestamps.append(data["timestamp"].numpy().copy())
                agent_ids.append(data["track_id"].numpy().copy())
                confs.append(fit_confidences.cpu().numpy().copy())
                if (len(confs) == 10): break
        write_pred_csv(f'{Global.MULTI_MODE_SUBMISSION}',
                timestamps=np.concatenate(timestamps),
                track_ids=np.concatenate(agent_ids),
                coords=np.concatenate(future_coords_offsets_pd),
                confs=np.concatenate(confs))