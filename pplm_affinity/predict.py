import os
import sys
import torch
import argparse
mian_path = os.path.dirname(__file__) + "/../"
sys.path.append(os.path.abspath(mian_path))

# import pplm_ppi
from pplm_affinity import PPLM_Affinity


def main():
    parser = argparse.ArgumentParser(description="Protein-Protein Binding Affinity Prediction",
                                     epilog="v0.0.1")

    parser.add_argument("receptor_seqs_path",
                        action="store",
                        help="Location of receptor sequences")

    parser.add_argument("ligand_seqs_path",
                        action="store",
                        help="Location of ligand sequences")

    parser.add_argument("output_path",
                        action="store",
                        help="Location of affinity score")

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
    models_path = [os.path.join(script_dir, "models/model_cv" + str(i) + ".pkl") for i in range(0, 5)]


    ### Read sequences ###
    seqA = read_sequence(args.receptor_seqs_path)
    seqB = read_sequence(args.ligand_seqs_path)

    ### Prediction ###
    with torch.no_grad():
        predictions_list = []
        for model_path in models_path:
            checkpoint = torch.load(model_path, map_location=device)
            pplm_model_param = checkpoint["pplm_param"]
            model_state = checkpoint["model_state_dict"]

            model = PPLM_Affinity(pplm_model_param)
            model.load_state_dict(model_state)
            model.to(device)

            predictions = model(seqA, seqB, device)
            predictions2 = model(seqB, seqA, device)
            predictions = (predictions + predictions2) / 2

            predictions_list.append(predictions)

        predictions = torch.stack(predictions_list)
        predictions = torch.mean(predictions, dim=0).squeeze().cpu().numpy()

        print("Predicted binding affinity:", predictions)

    with open(args.output_path, "w") as f:
        f.write(str(predictions))

def read_sequence(seq_path):
    seq = ""
    for line in open(seq_path, "r").readlines():
        if not line.startswith(">"):
            seq += line.strip()

    return seq

if __name__ == "__main__":
    main()
