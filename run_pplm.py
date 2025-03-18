import argparse
import pathlib
import pickle
import torch
from pplm import PPLM, Alphabet


def main():
    parser = argparse.ArgumentParser(
        description="PPLM: Protein-Protein Language Model."
    )
    parser.add_argument(
        "seqA_path",
        type=pathlib.Path,
        help="sequence file of the first sequence.",
    )
    parser.add_argument(
        "seqB_path",
        type=pathlib.Path,
        help="sequence file of the second sequence.",
    )

    parser.add_argument(
        "out_pkl_path",
        type=pathlib.Path,
        help="output path to storge the embedding and attention weights produced by PPLM",
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="gpu device specified",
    )

    args = parser.parse_args()

    ##### Loading PPLM Model #####
    alphabet = Alphabet.from_architecture()
    batch_converter = alphabet.get_batch_converter()
    model_data = torch.load("pplm/models/pplm_t33_650M.pt", map_location="cpu")
    model_param = model_data["param"]
    model_state = model_data["model"]

    model = PPLM(
        num_layers=model_param['encoder_layers'],
        embed_dim=model_param['encoder_embed_dim'],
        attention_heads=model_param['encoder_attention_heads'],
        token_dropout=False,
        alphabet=alphabet
    )

    assigned_device = "cuda:" + str(args.gpu_id)
    device = assigned_device if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.load_state_dict(model_state, strict=False)


    with torch.no_grad():
        seqA, seqB = '', ''
        for line in open(args.seqA_path, "r").readlines():
            if not line.startswith(">"):
                seqA += line.strip()
        for line in open(args.seqB_path, "r").readlines():
            if not line.startswith(">"):
                seqB += line.strip()
        lenA, lenB = len(seqA), len(seqB)


        seqA_labels, seqA_strs, seqA_tokens = batch_converter([('seqA', seqA)])
        seqB_labels, seqB_strs, seqB_tokens = batch_converter([('seqB', seqB)])
        tokens = torch.cat([seqA_tokens, seqB_tokens], dim=-1).to(device)

        inter_chain_mask = torch.ones((lenA + 2 + lenB + 2, lenA + 2 + lenB + 2), device=device)
        inter_chain_mask[:lenA + 2, :lenA + 2] = 0
        inter_chain_mask[lenA + 2:, lenA + 2:] = 0

        ##### running PPLM #####
        out = model(tokens, inter_chain_mask, repr_layers=[33], need_head_weights=True, return_contacts=False)

        ### Intra-protein embedding and attention weights, as well as inter-protein attention weights
        embed_A = out['representations'][33][0, 1:(lenA + 1), :].cpu().numpy()
        embed_B = out['representations'][33][0, -(lenB + 1):-1, :].cpu().numpy()
        attn_AA = out['attentions'].squeeze()[:, :, 1:(lenA + 1), 1:(lenA + 1)].reshape(20*33, lenA, lenA).cpu().numpy()
        attn_AB = out['attentions'].squeeze()[:, :, 1:(lenA + 1), -(lenB + 1):-1].reshape(20*33, lenA, lenB).cpu().numpy()
        attn_BA = out['attentions'].squeeze()[:, :, -(lenB + 1):-1, 1:(lenA + 1)].reshape(20*33, lenB, lenA).cpu().numpy()
        attn_BB = out['attentions'].squeeze()[:, :, -(lenB + 1):-1, -(lenB + 1):-1].reshape(20*33, lenB, lenB).cpu().numpy()

        inter_attn = (attn_AB + attn_BA.transpose(0, 2, 1)) / 2

        ### Full size of embedding and attention weights
        # embed = np.concatenate((embed_A, embed_B), axis=0)
        # attn = np.zeros((20*33, lenA + lenB, lenA + lenB))
        # attn[:, :lenA, :lenA] = attn_AA
        # attn[:, :lenA, lenA:] = attn_AB
        # attn[:, lenA:, :lenA] = attn_BA
        # attn[:, lenA:, lenA:] = attn_BB

        output = {
            'embed_A': embed_A,
            'embed_B': embed_B,
            'attn_AA': attn_AA,
            'attn_BB': attn_BB,
            'inter_attn': inter_attn,
            # 'embed': embed,
            # 'attn': attn,
        }

        with open(args.out_pkl_path, mode='wb') as fw:
            pickle.dump(output, fw)


if __name__ == "__main__":
    main()
