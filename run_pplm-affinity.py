import os
import torch
import argparse
from pplm_affinity import PPLM_Affinity

def main():
    parser = argparse.ArgumentParser(description="Protein-Protein Biniding Affinity Prediction",
                                     epilog="v0.0.1")

    parser.add_argument("receptor_seqs_path",
                        action="store",
                        help="Location of receptor sequences")

    parser.add_argument("ligand_seqs_path",
                        action="store",
                        help="Location of ligand sequences")

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

    cv_models_weight = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights/affinity_models.pkl"), map_location=device)

    model = PPLM_Affinity(cv_models_weight["pplm_param"])
    model.to(device)

    ### Read sequences ###
    seqA = read_sequence(args.receptor_seqs_path)
    seqB = read_sequence(args.ligand_seqs_path)

    ### Prediction ###
    with torch.no_grad():
        predictions_list = []
        for cv in range(0, 5):
            model.load_state_dict(cv_models_weight['cv' + str(cv)])
            predictions = model(seqA, seqB, device)
            predictions2 = model(seqB, seqA, device)
            predictions = (predictions + predictions2) / 2
            predictions_list.append(predictions)

        predictions = torch.stack(predictions_list)
        predictions = torch.mean(predictions, dim=0).squeeze().cpu().numpy()

        print("Predicted binding affinity:", predictions)

def read_sequence(seq_path):
    seq = ""
    for line in open(seq_path, "r").readlines():
        if not line.startswith(">"):
            seq += line.strip()

    return seq

if __name__ == "__main__":
    main()
    
