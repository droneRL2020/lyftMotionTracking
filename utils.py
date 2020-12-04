import torch
import numpy as np
import torch.nn.functional as F
from config.global_conf import Global

class Utils:
    @staticmethod
    def disassembling(data: list):   
        return torch.split(data, [3 for i in range(80)], 2)

    @staticmethod
    def disassembling_fitting(data: list):
        return torch.split(data, [3 for i in range(80)], 1)

    @staticmethod
    def average_weighted_norm(x, w):
        sum_w = torch.sum(w, dim=1).to(Global.DEVICE) # [batch_size, 1]
        sum_w_inv = torch.pow(torch.add(sum_w, torch.full(sum_w.size(), 1e-6 / 2.0).to(Global.DEVICE)), -1) # batch_size, 1, 1, 1]
        x_weighted = torch.mul(x, w) 
        x_weighted_sum = torch.sum(x_weighted, dim=1) 
        result = torch.mul(x_weighted_sum, sum_w_inv)
        result = torch.unsqueeze(result, dim=1) 
        return result

    @staticmethod
    def get_gaussian_mixture_model_from_independent_dists(samples_means: list, assignments: list):
        bs = assignments[0].size(0)
        num_of_modes = assignments[0].size(1)
        num_samples = len(samples_means)
        
        expanded_means = [torch.unsqueeze(samples_means[i], dim=1) for i in range(num_samples)]
        samples_means_concat = torch.cat(expanded_means, dim=1) # [batch_size, 20, 50, 3]
        
        expanded_assignments = [torch.unsqueeze(assignments[i], dim=1) for i in range(len(assignments))]
        assignments_adjusted = F.softmax(torch.cat(expanded_assignments, dim=1), dim=2) # [batch_size, 20, 3]
        # 각 hyps에 대한 모드별 conf
        
        mixture_weights = []
        means = []
        
        for k in range(num_of_modes):
            r_ik = assignments_adjusted[:,:,k].view(bs, -1, 1).to(Global.DEVICE)     # [batch_size, 20, 1]
            mu_k_x = Utils.average_weighted_norm(samples_means_concat[:,:,:,0], r_ik) # [batch_size, 20, 50],[batch_size, 20, 1]
            mu_k_y = Utils.average_weighted_norm(samples_means_concat[:,:,:,1], r_ik) # 위와 동일  
            mu_k_yaw = Utils.average_weighted_norm(samples_means_concat[:,:,:,2], r_ik) # 위와 동일
            
            w_k = torch.unsqueeze(torch.mean(r_ik, dim=1), dim=1)          # [batch_size, 1, 1]
            mu_k = torch.transpose(torch.cat([mu_k_x, mu_k_y, mu_k_yaw], dim=1), 1, 2) # [batch_size, 50, 3(x, y, yaw)]
            mixture_weights.append(w_k)
            means.append(mu_k)

        return means, mixture_weights
    
    @staticmethod
    def assemble_gmm_parameters_independent_dists(samples_means: list, assignments: list):
        means, mixture_weights =  Utils.get_gaussian_mixture_model_from_independent_dists(samples_means, assignments)
        return means, mixture_weights

    @staticmethod
    def map_writer_from_world_to_image(data, cfg):
    ######################################################################
    # input:  Tensor([batch_size, target_size, ])
    # output: Tensor[batch_size, target_size, 3]
    ######################################################################
    
        navs_rel_to_nav_in_world = torch.Tensor(data["target_positions"]) # (B,50,2)    
        bs,tl,_ = navs_rel_to_nav_in_world.shape
        centroid = data["centroid"][:,None,:].to(torch.float) # (B,2) -> (B,1,2)
        navs_in_world = centroid + navs_rel_to_nav_in_world # (B,1,2) + (B,50,2) -> (B,50,2)
        navs_in_world = torch.cat([navs_in_world, torch.ones((bs,tl,1))], dim=2) #(B,50,3)
        navs_in_world = navs_in_world.transpose(1,2) #(B,3,50)
        scale_image_tform_world = data["world_to_image"] #(B,3,3)
        ori_image_tform_world = scale_image_tform_world / 2 #(B,3,3)

        navs_in_image = torch.matmul(ori_image_tform_world.to(torch.float), navs_in_world) #(B,3,3) mamul (B,3,50) ->(B,3,50)
        navs_in_image *= 2
        navs_in_image = navs_in_image.transpose(1,2)[:,:,:2]  #B,50,2
        
        rs = cfg["raster_params"]["raster_size"]
        ec = cfg["raster_params"]["ego_center"]
        bias = torch.tensor([rs[0] * ec[0], rs[1] * ec[1]])[None,None,:] #1,1,2
        navs_in_image -= bias #B,50,2
        target_yaws = torch.Tensor(data["target_yaws"]) # B,50,1 
        nav_pose_in_image = torch.cat([navs_in_image, target_yaws], dim=2)  #B,50,3
        nav_pose_in_image[:,:,1:] *= -1 # bias 뺀다음 y, yaw를 뒤집어준담에 다시 바이아스 더하기
        nav_pose_in_image[:,:,0] += 56 
        nav_pose_in_image[:,:,1] += 112  # B,50,3
        return nav_pose_in_image #B,50,3

    @staticmethod
    def map_writer_from_image_to_world(data, nav_pose_in_image, cfg):
        ''' nav_pose_in_image[B,50,3], should loop each mode'''
        nav_pose_in_image[:,:,0] -= 56
        nav_pose_in_image[:,:,1] -= 112
        nav_pose_in_image[:,:,1:] /= -1
        result_yaw = nav_pose_in_image[:,:,2]
        
        rs = cfg["raster_params"]["raster_size"]
        ec = cfg["raster_params"]["ego_center"]
        bias = torch.tensor([rs[0] * ec[0], rs[1] * ec[1]])[None,None,:].to(Global.DEVICE) #1,1,2
        nav_pose_in_image[:,:,:2] += bias  # B,50,2
        navs_in_image = nav_pose_in_image[:,:,:2] / 2 # B,50,2
        navs_in_image = torch.cat([navs_in_image, torch.ones((navs_in_image.size(0), 50,1)).to(Global.DEVICE)], dim=2) #(B,50,3)
        
        scale_image_tform_world = data["world_to_image"] #(B,3,3)
        ori_image_tform_world = scale_image_tform_world / 2 #(B,3,3)
        ori_world_tform_image = Utils.inverse(ori_image_tform_world) #(B,3,3)
        navs_in_world = torch.matmul(ori_world_tform_image.to(Global.DEVICE), torch.transpose(navs_in_image,2,1).to(Global.DEVICE)) #(B,3,3) matmul (B,3,50) -> (B,3,50)
        navs_in_world = torch.transpose(navs_in_world, 2,1) # (B,3,50) -> (B,50,3)
        centroid = data["centroid"][:,None,:].to(torch.float).to(Global.DEVICE)  #(B,2) -> (B,1,2)
        navs_in_world[:,:,:2] -= centroid # (B,50,2) - (B,1,2) -> (B,50,2)
        
        return navs_in_world[:,:,:2] #(B,50,2)

    @staticmethod
    def inverse(a_tform_b):
        '''
        a_tform_b : [B,3,3]
        2x2 is rotational matrix, 1,2 is translational vector
        '''
        bs, _, _ = a_tform_b.shape
        a_tform_b = a_tform_b.cpu().numpy()
        b_rot_a = torch.Tensor(a_tform_b[:,:2,:2].transpose(0,2,1)) #[B,2,2] -> [B,2,2]
        b_trans_a = -torch.matmul(torch.Tensor(a_tform_b[:,:2,:2].transpose(0,2,1)), torch.Tensor(a_tform_b[:,:2,2]).unsqueeze(2)) # [B,2,1]
        b_tform_a = torch.cat([b_rot_a,b_trans_a],dim=2) #[B,2,3]
        temp = torch.zeros([bs,3,3]) #[B,3,3]
        for i in range(bs):
            temp[i,:,:] = torch.cat([b_tform_a[i,:,:], torch.Tensor([[0,0,1]])], dim=0)  #[2,3] cat [1,3] -> [3,3]
        b_tform_a = temp #[B,3,3]
        return b_tform_a #[B,3,3]