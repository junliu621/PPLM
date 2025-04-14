import torch
from torch import nn

class PPLM_PPI(nn.Module):
    def __init__(self, attn_dim=660, embed_dim=1280):
        super(PPLM_PPI, self).__init__()
        prev_dim = attn_dim
        self.linear_intra = nn.Linear(attn_dim, prev_dim)
        self.linear_inter = nn.Linear(attn_dim, prev_dim)
        self.linear_embed = nn.Linear(embed_dim, prev_dim)

        layers = []
        layers.append(nn.Linear(prev_dim * 10, 1024))
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

    def forward(self, inter_attn, intra_attn_A, intra_attn_B, embed_A, embed_B, inter_attn2, intra_attn_A2, intra_attn_B2, embed_A2, embed_B2):
        inter_attn = self.linear_inter(inter_attn)
        intra_attn_A = self.linear_intra(intra_attn_A)
        intra_attn_B = self.linear_intra(intra_attn_B)
        embed_A = self.linear_embed(embed_A)
        embed_B = self.linear_embed(embed_B)
        inter_attn2 = self.linear_inter(inter_attn2)
        intra_attn_A2 = self.linear_intra(intra_attn_A2)
        intra_attn_B2 = self.linear_intra(intra_attn_B2)
        embed_A2 = self.linear_embed(embed_A2)
        embed_B2 = self.linear_embed(embed_B2)

        features = torch.cat([inter_attn, intra_attn_A, intra_attn_B, embed_A, embed_B, inter_attn2, intra_attn_A2, intra_attn_B2, embed_A2, embed_B2], dim=0)  # for  batchsize > 1

        preds = self.mlp(features)

        return preds
