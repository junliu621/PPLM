import torch
from torch import nn
import os, sys



class PPLM_PPI(nn.Module):
    def __init__(self, input_dim=660, embedding_dim=1280):
        super(PPLM_PPI, self).__init__()
        prev_dim = input_dim
        self.linear_intra = nn.Linear(prev_dim, prev_dim)
        self.linear_inter = nn.Linear(prev_dim, prev_dim)
        self.linear_embed = nn.Linear(embedding_dim, prev_dim)

        layers = []
        layers.append(nn.Linear(prev_dim * 5, 1024))
        layers.append(nn.LayerNorm(1024))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(1024, 512))
        layers.append(nn.LayerNorm(512))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(512, 256))
        layers.append(nn.LayerNorm(256))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 128))
        layers.append(nn.LayerNorm(128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 1))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

    def forward(self, inter_attn, intra_attn_A, intra_attn_B, embed_A, embed_B):
        inter_attn = self.linear_inter(inter_attn)
        intra_attn_A = self.linear_intra(intra_attn_A)
        intra_attn_B = self.linear_intra(intra_attn_B)
        embed_A = self.linear_embed(embed_A)
        embed_B = self.linear_embed(embed_B)

        # print(inter_attn.shape, intra_attn_A.shape, intra_attn_B.shape, embed_A.shape, embed_B.shape)

        features = torch.cat([inter_attn, intra_attn_A, intra_attn_B, embed_A, embed_B], dim=-1)   #for  batchsize > 1

        preds = self.mlp(features)  # Shape: (batch_size, 1)

        return preds
