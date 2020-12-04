import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from typing import Dict
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer

from config.global_conf import Global
from net import TopkNet
from utils import Utils
from loss import Loss

class Trainer():
    def __init__(self, W_PATH=None):
        os.environ["L5KIT_DATA_FOLDER"] = f"{Global.DIR_INPUT}/l5kit/dataset"
        self.cfg = Global.getConfig()
        self.train_cfg = self.cfg["train_data_loader"]
        dm = LocalDataManager(None)
        rasterizer = build_rasterizer(self.cfg, dm)
        train_zarr = ChunkedDataset(dm.require(self.train_cfg["key"])).open()
        self.train_dataset = AgentDataset(self.cfg, train_zarr, rasterizer)
        self.straight_train_dataloader = DataLoader(self.train_dataset,
                              shuffle=True,
                              batch_size=32,
                              num_workers=self.train_cfg["num_workers"])
        self.W_PATH = W_PATH

    def run(self):
        model = TopkNet(self.cfg).to(Global.DEVICE)
        Global.load_weight(model, W_PATH)
        optimizer = optim.Adam(model.parameters(), lr=1e-6)
        Global.load_weight(model, self.W_PATH)
        progress_bar = tqdm(range(self.cfg["train_params"]["max_num_steps"]))
        losses_train = []
        prelosses_train = []

        straight_it = iter(self.straight_train_dataloader)
        
        for itr in progress_bar:
            data = next(straight_it)
            model.train()
            torch.set_grad_enabled(True)
            
            inputs = data["image"].to(Global.DEVICE) 
            targets = Utils.map_writer_from_world_to_image(data, self.cfg).to(Global.DEVICE)
            target_availabilities = data["target_availabilities"].to(Global.DEVICE)
            samples_means, means, mixture_weights = model(inputs)

            fit_outputs = torch.stack([mean for mean in means], dim=1)
            fit_confidences = torch.stack([mixture_weight for mixture_weight in mixture_weights], dim=1).squeeze()

            if(itr <= 0):
                if(itr <= 20):
                    loss = Loss.sampling_loss(targets, samples_means, target_availabilities)
                elif(itr <= 40):
                    loss = Loss.sampling_loss(targets, samples_means, target_availabilities, "epe-top-n", 40)
                elif(itr <= 60):
                    loss = Loss.sampling_loss(targets, samples_means, target_availabilities, "epe-top-n", 20)
                elif(itr <= 80):
                    loss = Loss.sampling_loss(targets, samples_means, target_availabilities, "epe-top-n", 10)
                elif(itr <= 100):
                    loss = Loss.sampling_loss(targets, samples_means, target_availabilities, "epe-top-n", 5)
                elif(itr <= 120):
                    loss = Loss.sampling_loss(targets, samples_means, target_availabilities, "epe")
                for i, param in enumerate(model.parameters()):
                    if i >= 68:
                        param.requires_grad = False
                
            elif(itr <= 200):
                loss, pre_loss = Loss.fitting_loss(targets, fit_outputs, fit_confidences, target_availabilities)     
                for i, param in enumerate(model.parameters()):
                    if i >= 68:
                        param.requires_grad = True
                    elif i < 68:
                        param.requires_grad = False
            elif(itr <= 300):
                loss, pre_loss = Loss.fitting_loss(targets, fit_outputs, fit_confidences, target_availabilities)
                for i, param in enumerate(model.parameters()):
                    param.requires_grad = True

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_train.append(loss.item())

            if (itr <= 100):
                progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train[-100:])}")
            else:
                prelosses_train.append(pre_loss.item())
                progress_bar.set_description(f"pre_loss: {pre_loss.item()} loss(avg): {np.mean(prelosses_train[-100:])}")

            if (itr+1) % self.cfg['train_params']['checkpoint_every_n_steps'] == 0 and not Global.DEBUG:
                torch.save(model.state_dict(), f"./output/param/topk_model_state_{itr}.pth")