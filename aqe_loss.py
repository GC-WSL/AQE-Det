import torch.nn as nn
import torch
from ..builder import ROTATED_LOSSES
import math
import numpy as np
import torch.nn.functional as F
import ipdb



@ROTATED_LOSSES.register_module()
class aqe_loss(nn.Module):
    # input: [angle sigma], gt_angle
    def __init__(self, loss_weight=1.0):
        super(aqe_loss, self).__init__()
        self.loss_weight = loss_weight
        #self.criterion = nn.KLDivLoss(size_average=False)
        self.ce = nn.CrossEntropyLoss(size_average=False, reduce=False)
        #self.focal = HeadFocalLossAngle(num_class=180, size_average=True)
        self.mse = nn.MSELoss(reduce=False,size_average=False)
        #self.sinkhorn = SinkhornDistance(eps=0.1,max_iter=100,reduction=None)

    def forward(self, angle, sigma, target, weight, *args, **kwargs): 
        ###################angle######################
        angle_target = target + math.pi/2  #0-pi
        angle = torch.sigmoid(angle)*math.pi  #0-pi
        ori_sigma = torch.sigmoid(sigma)
        angle_weight = weight
        avg_factor = torch.sum(angle_weight > 0).float().item() 
        
        if avg_factor > 0:
            gauss_x = torch.from_numpy((np.array(range(0, 180, 1))/180)*math.pi).cuda().float()
            gauss_x = gauss_x.repeat(int(avg_factor)).reshape(-1,180).cuda().float()
            angle = angle[angle_weight > 0].float()
            angle_label = angle.repeat(180).reshape(180,-1).permute(1,0).cuda().float()
            ori_sigma = ori_sigma[angle_weight>0].float()
            sigma = ori_sigma.repeat(180).reshape(180,-1).permute(1,0).cuda().float()
            angle_target = angle_target[angle_weight > 0].cuda().float()
            angle_weight = angle_weight[angle_weight>0].float()
            gauss_label = torch.exp(-((gauss_x-angle_label)**2)/(2*sigma**2+ 1e-8))
            dirac_label = torch.zeros(int(avg_factor),180).cuda().scatter_(1,(180*angle_target/math.pi).long().reshape(-1,1),1).cuda().float()
            # logit = F.softmax(gauss_label, dim=1)
            # gamma=5*(1-0.5*angle_weight)
            # weight2 = torch.pow((1-torch.sum(logit*dirac_label,1)), gamma)
            loss_angle = self.ce(gauss_label, (180*(angle_target/math.pi)).long().reshape(-1))*angle_weight - 0.8#*weight2
        else:
            loss_angle = torch.abs(angle-angle_target)* angle_weight.float()

        loss_angle = self.loss_weight * (torch.sum(loss_angle))[None] / (avg_factor+1)  
        return loss_angle
