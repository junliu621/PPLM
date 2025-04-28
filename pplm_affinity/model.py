import torch
from torch import nn
import os, sys




class PPLM_Affinity(nn.Module):
    def __init__(self, model_param, attn_dim=660, embed_dim=1280):
        super(PPLM_Affinity, self).__init__()

        ################################### Define PPLM model #####################################
        mian_path = os.path.dirname(__file__) + "/../"
        sys.path.append(os.path.abspath(mian_path))
        from pplm import PPLM, Alphabet

        ##### Loading PPLM Model #####
        self.alphabet = Alphabet.from_architecture()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model_param = model_param

        self.model = PPLM(
            num_layers=self.model_param['encoder_layers'],
            embed_dim=self.model_param['encoder_embed_dim'],
            attention_heads=self.model_param['encoder_attention_heads'],
            token_dropout=False,
            alphabet=self.alphabet
        )

        for name, param in self.model.named_parameters():
            if 'layers.32.' not in name:
                param.requires_grad = False

        #########################################################################################
        self.predict_module = nn.Sequential(
            nn.Linear(self.model_param['encoder_embed_dim'], 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def load_init_pplm_model_state(self, model_state):
        self.model.load_state_dict(model_state, strict=False)

    def forward(self, seqA, seqB, device):
        lenA, lenB = len(seqA), len(seqB)
        seqA_labels, seqA_strs, seqA_tokens = self.batch_converter([('seqA', seqA)])
        seqB_labels, seqB_strs, seqB_tokens = self.batch_converter([('seqB', seqB)])
        tokens = torch.cat([seqA_tokens, seqB_tokens], dim=-1).to(device)

        inter_chain_mask = torch.ones((len(seqA) + 2 + len(seqB) + 2, len(seqA) + 2 + len(seqB) + 2), device=device)
        inter_chain_mask[:len(seqA) + 2, :len(seqA) + 2] = 0
        inter_chain_mask[len(seqA) + 2:, len(seqA) + 2:] = 0

        ##### running PPLM #####
        out = self.model(tokens, inter_chain_mask, repr_layers=[33], need_head_weights=True, return_contacts=False)

        embed_A = out['representations'][33][0, 1:(lenA + 1), :]
        embed_B = out['representations'][33][0, -(lenB + 1):-1, :]
        embed = torch.cat([embed_A, embed_B], dim=0)
        del embed_A, embed_B, out

        max_pooling_embed = torch.max(embed, dim=0, keepdim=False).values  # 计算最大池化
        prediction = self.predict_module(max_pooling_embed)

        return prediction
