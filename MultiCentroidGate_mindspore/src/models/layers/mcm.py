from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import math

class DifferMCM(nn.Module):
    def __init__(self, in_channel, expansion_factor, reuse=True) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.mcm = nn.ParameterList()
        self.expansion_factor = expansion_factor 
        self.centroids_task = []
        self.reuse = reuse
    
    @property
    def out_dim(self):
        return np.sum(self.centroids_task)
    
    def add_classes(self, n_classes): 
        nb_centroid = math.ceil(n_classes * self.expansion_factor)
        self.centroids_task.append(nb_centroid)

        mcm = nn.Parameter(torch.empty(nb_centroid, self.in_channel))
        nn.init.kaiming_normal_(mcm.data, nonlinearity="linear") 
        assert isinstance(mcm, nn.Parameter) and mcm.requires_grad == True
        self.mcm.append(mcm)
        if not self.reuse:
            for i in self.mcm[:-1]:
                nn.init.kaiming_normal_(i.data, nonlinearity="linear") 
 
    def forward(self, feature, sim_fn):
        results, amax_result = [], []
        for m in self.mcm:
            d = sim_fn(feature, m)
            results.append(d)
            amax_result.append(d.amax(1, keepdim=True)) 
        return torch.cat(results, 1), torch.cat(amax_result, 1)
    
    def clone(self):
        return self