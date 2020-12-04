import numpy as np
import torch
from config.global_conf import Global
from torch import Tensor

class Loss:
    @staticmethod
    def sampling_loss(gt, samples_means, avails, mode="epe-all", top_n=80):
        num_samples = len(samples_means)
        gts = torch.stack([gt for _ in range(num_samples)], dim=1) # (B,20,50,3)
        samples_means = torch.stack([sample_means for sample_means in samples_means], dim=1) # (B,20,50,3)
        avails = avails[:,None,:,None] #(B,1,50,1)
        eps = 0.001
        diff = torch.pow(torch.mul((gts-samples_means), avails), 2) #(B,40,50,3)
        channels_sum = torch.sum(diff, dim=3) # (B,40,50)   
        dis_factor = 0.98
        alpha = torch.FloatTensor([dis_factor**(49-i) for i in range(50)]).view(1, 1, 50).to(Global.DEVICE)
        channels_sum = alpha*channels_sum
        
        time_sum = torch.sum(channels_sum, dim=2) #(B,40)   
        spatial_epes = torch.sqrt(time_sum + eps) #(B,40)
        sum_losses = torch.tensor(0.0)
        
        if(mode == "epe-all"):
            # hyps : 20개 다씀
            for i in range(num_samples):
                loss = torch.mul(torch.mean(spatial_epes[:, i]), 1.0).to(Global.DEVICE)  # 각 hyps(나중에모드)에대해 16개 배치 평균 내고 소수로 만들어줌
                sum_losses = torch.add(loss, sum_losses).to(Global.DEVICE)
        elif(mode == "epe-top-n") and top_n > 1:
            # hyps : 10 -> 5 -> 3
            top_k, indices = torch.topk(-spatial_epes, top_n)  # topk 는 어짜피 (16,40) 에서 40을 인덱싱하니까 걍 써도됨
            spatial_epes_min = -top_k
            for i in range(top_n):
                loss = torch.mul(torch.mean(spatial_epes_min[:, i]), 1.0).to(Global.DEVICE)  # 각 모드에대해 16개 배치 평균 내고 소수로 만들어줌
                sum_losses = torch.add(loss, sum_losses).to(Global.DEVICE)
        elif(mode == "epe"):
            # hyps : 40개중 1개만 씀
            spatial_epe, indices = torch.min(spatial_epes, dim=1) # spaital_epe (B,)
            loss = torch.mul(torch.mean(spatial_epe),1.0).to(Global.DEVICE)
            sum_losses = torch.add(loss, sum_losses).to(Global.DEVICE)
            
        return sum_losses
    
    @staticmethod
    def fitting_loss(gt: Tensor, pred, confidences: Tensor, avails: Tensor):
        batch_size, num_modes, future_len, num_coords = pred.shape
        gt = torch.unsqueeze(gt, 1)  # add modes
        avails = avails[:, None, :, None]  # add modes and cords

        error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)
        with np.errstate(divide="ignore"):  
            dis_factor = 0.98
            alpha = torch.FloatTensor([dis_factor**i for i in range(50)]).view(1, 1, 50).to(Global.DEVICE)
            pre_error = error
            pre_error = torch.log(confidences) - 0.5 * torch.sum(pre_error, dim=-1)
            error = alpha * error
            error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

        pre_max_value, _ = pre_error.max(dim=1, keepdim=True)
        max_value, _ = error.max(dim=1, keepdim=True)  

        pre_error = -torch.log(torch.sum(torch.exp(pre_error - pre_max_value), dim=-1, keepdim=True)) - pre_max_value 
        error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
        return torch.mean(error), torch.mean(pre_error)