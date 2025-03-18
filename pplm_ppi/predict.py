import os
import sys
import torch
import argparse
mian_path = os.path.dirname(__file__) + "/../"
sys.path.append(os.path.abspath(mian_path))

# import pplm_ppi
from pplm_ppi import PPLM_PPI


def main():
    parser = argparse.ArgumentParser(description="Protein-Protein Interaction Prediction",
                                     epilog="v0.0.1")

    parser.add_argument("seqA_path",
                        action="store",
                        help="Location of sequence A")

    parser.add_argument("seqB_path",
                        action="store",
                        help="Location of sequence B")

    parser.add_argument("--gpu_id",
                        "-gpu",
                        type=int,
                        default=0,
                        help="gpu device specified",
                        )

    args = parser.parse_args()

    ### Define model ###
    assigned_device = "cuda:" + str(args.gpu_id)
    device = assigned_device if torch.cuda.is_available() else "cpu"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_path = [os.path.join(script_dir, "models/model" + str(i) + ".pkl") for i in range(1, 6)]

    model = PPLM_PPI()
    model.to(device)

    ### Read sequences ###
    seqA = read_sequence(args.seqA_path)
    seqB = read_sequence(args.seqB_path)

    ### Get pplm features ###
    mean_inter_attn, mean_attn_AA, mean_attn_BB, mean_embed_A, mean_embed_B, max_inter_attn, max_attn_AA, max_attn_BB, max_embed_A, max_embed_B = get_pplm_features(seqA, seqB, device)

    ### Prediction ###
    with torch.no_grad():
        predictions_list = []
        for model_path in models_path:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["net"])

            predictions = model(mean_inter_attn, mean_attn_AA, mean_attn_BB, mean_embed_A, mean_embed_B, max_inter_attn, max_attn_AA, max_attn_BB, max_embed_A, max_embed_B)
            predictions_list.append(predictions)

        predictions = torch.stack(predictions_list)
        predictions = torch.mean(predictions, dim=0).squeeze().cpu().numpy()

        print("Predicted Interaction probability:", predictions)


def get_pplm_features(seqA, seqB, device):
    mian_path = os.path.dirname(__file__) + "/../"
    sys.path.append(os.path.abspath(mian_path))

    from pplm import PPLM, Alphabet
    model_location = os.path.join(mian_path, 'pplm/models/', 'pplm_t33_650M.pt')

    ##### Loading PPLM Model #####
    alphabet = Alphabet.from_architecture()
    batch_converter = alphabet.get_batch_converter()
    model_data = torch.load(model_location, map_location="cpu")
    model_param = model_data["param"]
    model_state = model_data["model"]

    model = PPLM(
        num_layers=model_param['encoder_layers'],
        embed_dim=model_param['encoder_embed_dim'],
        attention_heads=model_param['encoder_attention_heads'],
        token_dropout=False,
        alphabet=alphabet
    )
    model.to(device)
    model.load_state_dict(model_state, strict=False)

    with torch.no_grad():
        seqA_labels, seqA_strs, seqA_tokens = batch_converter([('seqA', seqA)])
        seqB_labels, seqB_strs, seqB_tokens = batch_converter([('seqB', seqB)])
        tokens = torch.cat([seqA_tokens, seqB_tokens], dim=-1).to(device)

        inter_chain_mask = torch.ones((len(seqA) + 2 + len(seqB) + 2, len(seqA) + 2 + len(seqB) + 2), device=device)
        inter_chain_mask[:len(seqA) + 2, :len(seqA) + 2] = 0
        inter_chain_mask[len(seqA) + 2:, len(seqA) + 2:] = 0

        ##### running PPLM #####
        out = model(tokens, inter_chain_mask, repr_layers=[33], need_head_weights=True, return_contacts=False)

        embed_A = out['representations'][33][0, 1:(len(seqA) + 1), :]
        embed_B = out['representations'][33][0, -(len(seqB) + 1):-1, :]

        attn_AA = out['attentions'].squeeze()[:, :, 1:(len(seqA) + 1), 1:(len(seqA) + 1)].reshape(33 * 20, len(seqA), len(seqA))
        attn_AB = out['attentions'].squeeze()[:, :, 1:(len(seqA) + 1), -(len(seqB) + 1):-1].reshape(33 * 20, len(seqA), len(seqB))
        attn_BA = out['attentions'].squeeze()[:, :, -(len(seqB) + 1):-1, 1:(len(seqA) + 1)].reshape(33 * 20, len(seqB), len(seqA))
        attn_BB = out['attentions'].squeeze()[:, :, -(len(seqB) + 1):-1, -(len(seqB) + 1):-1].reshape(33 * 20, len(seqB), len(seqB))
        inter_attn = (attn_AB + attn_BA.transpose(1, 2)) / 2

        mean_inter_attn = inter_attn.mean(dim=[1, 2])
        mean_attn_AA = attn_AA.mean(dim=[1, 2])
        mean_attn_BB = attn_BB.mean(dim=[1, 2])
        mean_embed_A = embed_A.mean(dim=[0])
        mean_embed_B = embed_B.mean(dim=[0])
        max_inter_attn = torch.amax(inter_attn, dim=(1, 2))
        max_attn_AA = torch.amax(attn_AA, dim=(1, 2))
        max_attn_BB = torch.amax(attn_BB, dim=(1, 2))
        max_embed_A = torch.amax(embed_A, dim=0)
        max_embed_B = torch.amax(embed_B, dim=0)

        return mean_inter_attn, mean_attn_AA, mean_attn_BB, mean_embed_A, mean_embed_B, max_inter_attn, max_attn_AA, max_attn_BB, max_embed_A, max_embed_B

def read_sequence(seq_path):
    seq = ""
    for line in open(seq_path, "r").readlines():
        if not line.startswith(">"):
            seq += line.strip()

    return seq

if __name__ == "__main__":
    main()