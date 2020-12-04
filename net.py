import torch
from utils import Utils
from torch import nn, optim
from torchvision.models.resnet import resnet18

class TopkNet(nn.Module):
    def __init__(self, cfg, num_modes=3):
        super().__init__()
        self.backbone = resnet18(pretrained=True, progress=True)
        
        # history가10개인경우 num_in_channels는 25. 25는 rasterized image 의 채널
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        
        ## SAMPLING NETWORK
        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )
        
        backbone_out_features = 512
        

        # You can add more layers here.
        self.sample_fcs = nn.Sequential(
            nn.Linear(in_features=backbone_out_features, out_features=2048),
            nn.Linear(2048, 4096)
        )
        # hypothesis * future_len * (x,y,yaw) * 2
        self.sample_logit = nn.Linear(in_features=4096, out_features=80*50*3) 
        
        ## FITTING NETWORK
        self.fitting_fcs = nn.Sequential(
            nn.Linear(80*50*3, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU()
        )
        
        self.fit_logit = nn.Linear(4096, 80*3) #20개 [x,y,yaw] set에 대한 mode 3개에 대한 모델 assignment 나온거
        
        # X, Y, yaw coords for the future positions [output shape: B x 50 x 3]
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 3 * self.future_len # 150
        self.num_modes = num_modes
        self.num_preds = num_targets * self.num_modes # 450
        
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)  # shape: [batch_size, 512]
        bs = x.size(0)
        
        # Sampling
        x = self.sample_fcs(x)
        samples_logit = self.sample_logit(x)                     # [batch_size, 20*50*3]
        samples_out = samples_logit.view(bs, 50, -1)              # [batch_size, 50, 20*3]
        samples_means = Utils.disassembling(samples_out) # [batch_size, 50, 3] 각각 20개씩 tuple로 되어있음 

        x_2 = samples_logit  # [batch_size, 50*20*3]
        # Fitting
        x_2 = self.fitting_fcs(x_2)     # [batch_size, 1024]
        predicted = self.fit_logit(x_2) # [batch_size, 20*3]
        
        out_soft_assignments = Utils.disassembling_fitting(predicted) # [batch_size, 50, 3] 짜리 20개 담긴 tuple
        means, mixture_weights = Utils.assemble_gmm_parameters_independent_dists(samples_means=samples_means,
                                                                                               assignments=out_soft_assignments)
        
        return samples_means, means, mixture_weights