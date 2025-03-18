import os
import sys
import torch
import torch.nn as nn
mian_path = os.path.dirname(__file__) + "/../"
sys.path.append(os.path.abspath(mian_path))

from pplm_contact.Module import *

class PPLM_Contact(nn.Module):

    def __init__(self,
                 intra_1d_dim=768+20,
                 intra_2d_dim=144+2+64,
                 inter_2d_dim=144+2+660,
                 channels=64,
                 num_blocks=12,
                 droupout=0.10,
                 ):
        super(PPLM_Contact, self).__init__()
        self.intra_1d_dim = intra_1d_dim
        self.intra_2d_dim = intra_2d_dim
        self.inter_2d_dim = inter_2d_dim
        self.channels = channels
        self.num_blocks = num_blocks
        self.droupout = droupout

        ### ResNet
        self.intra_resnet = Intra_ResNet(dim_1d=self.intra_1d_dim, dim_2d=self.intra_2d_dim, channels=self.channels)
        self.inter_resnet = Inter_ResNet(dim_2d=self.inter_2d_dim, channels=self.channels)

        ### Transformer
        self.InterTriangleMulti_R = nn.ModuleList([InterTriangleMultiplication_S(channel_z=self.channels, channel_c=self.channels, transpose=False) for _ in range(num_blocks)])
        self.InterTriangleMulti_L = nn.ModuleList([InterTriangleMultiplication_S(channel_z=self.channels, channel_c=self.channels, transpose=True) for _ in range(num_blocks)])
        self.InterCrossAttn_R = nn.ModuleList([InterCrossAttention_S(channel_z=self.channels, bias=True, transpose=False) for _ in range(num_blocks)])
        self.InterCrossAttn_L = nn.ModuleList([InterCrossAttention_S(channel_z=self.channels, bias=True, transpose=True) for _ in range(num_blocks)])
        self.InterSelfAttn_R = nn.ModuleList([InterSelfAttention_S(channel_z=self.channels, bias=False, transpose=False) for _ in range(num_blocks)])
        self.InterSelfAttn_L = nn.ModuleList([InterSelfAttention_S(channel_z=self.channels, bias=False, transpose=True) for _ in range(num_blocks)])
        self.Transition = nn.ModuleList([Transition(channel_z=self.channels) for _ in range(num_blocks)])
        self.drop = nn.Dropout(self.droupout)

        self.norm_final = nn.LayerNorm(self.channels)
        self.Linear_final = nn.Linear(self.channels, 1)
        self.act_final = nn.Sigmoid()

    def forward(self, intra1_1d, intra1_2d, intra2_1d, intra2_2d, inter_2d, intra1_dist=None, intra2_dist=None, last_layer=False):

        intra1_2d = self.intra_resnet(intra1_1d, intra1_2d)
        intra2_2d = self.intra_resnet(intra2_1d, intra2_2d)
        inter_2d = self.inter_resnet(inter_2d)

        intra1_2d = intra1_2d.permute(0, 2, 3, 1)
        intra2_2d = intra2_2d.permute(0, 2, 3, 1)
        inter_2d = inter_2d.permute(0, 2, 3, 1)


        for block in range(self.num_blocks):
            inter_2d = inter_2d + self.drop(self.InterTriangleMulti_R[block](inter_2d, intra1_2d)) + self.drop(self.InterTriangleMulti_L[block](inter_2d, intra2_2d, transpose=True))
            inter_2d = inter_2d + self.drop(self.InterCrossAttn_R[block](inter_2d, intra1_2d)) + self.drop(self.InterCrossAttn_L[block](inter_2d, intra2_2d, transpose=True))
            inter_2d = inter_2d + self.drop(self.InterSelfAttn_R[block](inter_2d, intra2_dist)) + self.drop(self.InterSelfAttn_L[block](inter_2d, intra1_dist, transpose=True))
            inter_2d = inter_2d + self.Transition[block](inter_2d)

        inter_2d = self.norm_final(inter_2d)
        if last_layer:
            representation = inter_2d.permute(0,3,1,2)
        inter_2d = self.Linear_final(inter_2d)
        inter_contact = self.act_final(inter_2d)

        if last_layer:
            return inter_contact.permute(0,3,1,2)[0], representation

        return inter_contact.permute(0,3,1,2)[0]
