import torch
import torch.nn as nn
from lib.config import cfg
from lib.utils import trans_utils
import numpy as np

class PRSNet(nn.Module):
    def __init__(self,):
        super(PRSNet, self).__init__()
        conv = []
        activation = nn.LeakyReLU(0.2, True)
        input_ch = cfg.network.input_ch
        output_ch = cfg.network.output_ch

        biasTerms = {}
        biasTerms['plane1']=[1,0,0,0]
        biasTerms['plane2']=[0,1,0,0]
        biasTerms['plane3']=[0,0,1,0]
        biasTerms['quat1']=[0, 0, 0, np.sin(np.pi/2)]
        biasTerms['quat2']=[0, 0, np.sin(np.pi/2), 0]
        biasTerms['quat3']=[0, np.sin(np.pi/2), 0, 0]

        for i in range(cfg.network.conv_layer):
            conv += [nn.Conv3d(input_ch, output_ch, cfg.network.kernel_size, stride=1, padding=1)]
            conv += [nn.MaxPool3d(2), activation]
            input_ch = output_ch
            output_ch *= 2
        self.conv = nn.Sequential(*conv)

        self.planeLayers = nn.ModuleList()
        for i in range(cfg.task_arg.num_plane):
            planeLayer = [
                nn.Linear(input_ch, input_ch // 2), 
                activation, 
                nn.Linear(input_ch // 2, input_ch // 4), 
                activation
            ]
            last = nn.Linear(input_ch // 4, 4)
            last.weight.data = torch.zeros(4, input_ch // 4)
            last.bias.data = torch.Tensor(biasTerms['plane'+str(i+1)])
            planeLayer += [last]
            self.planeLayers.append(nn.Sequential(*planeLayer))

        self.quatLayers = nn.ModuleList()
        for i in range(cfg.task_arg.num_quat):
            quatLayer = [
                nn.Linear(input_ch, input_ch // 2),
                activation,
                nn.Linear(input_ch // 2, input_ch // 4), 
                activation
            ]
            last = nn.Linear(input_ch // 4, 4)
            last.bias.data = torch.Tensor(biasTerms['quat'+str(i+1)])
            quatLayer += [last]
            self.quatLayers.append(nn.Sequential(*quatLayer))

    def forward(self, input):
        feature = self.conv(input)
        feature = feature.view(feature.size(0), -1)
        plane = []
        quat = []

        for i in range(cfg.task_arg.num_plane):
            planeLayer = self.planeLayers[i]
            plane += [trans_utils.normalize(planeLayer(feature), 3)]
        
        for i in range(cfg.task_arg.num_quat):
            quatLayer = self.quatLayers[i]
            quat += [trans_utils.normalize(quatLayer(feature))]

        return plane, quat