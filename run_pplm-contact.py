import os
import sys
import pathlib
import numpy as np
import pickle
import torch
import subprocess
import argparse
from pplm_contact.model import PPLM_Contact
from pplm_contact.utils import extract_seq_and_dist_map, pairing_msa, RBF
from pplm_contact.LoadHHM import load_hmm
from pplm_contact.config import *

from pplm_contact import *
# from pplm_contact import define_param, extract_seq_and_dist_map, pairing_msa, extract_MSA_features, get_pplm_features, collect_all_features, predict_contact
def main():
    parser = argparse.ArgumentParser(description="Protein-Protein Contact Prediction",
                                     epilog="v0.0.1")

    parser.add_argument("pdbA_path",
                        type=pathlib.Path,
                        help="Location of pdb A")

    parser.add_argument("pdbB_path",
                        type=pathlib.Path,
                        help="Location of pdb B")

    parser.add_argument("output_folder",
                        type=pathlib.Path,
                        help="Location to store output files")

    parser.add_argument("--n_cpu",
                        type=int,
                        default=8,
                        help="Number of CPU cores for search MSA",
                        )

    parser.add_argument("--gpu_id",
                        type=int,
                        default=0,
                        help="gpu device specified",
                        )

    args = parser.parse_args()

    ### Define parameters ###
    define_param(args)

    ##### Step 0: Process the input pdb (clean pdb & extract sequence & derive monomer distance map) #####
    print("========== Step 0: Processing the intput pdb (Clean pdb, extract seq and dist map) ==========")

    subprocess.run("grep \"^ATOM\" " + str(args.pdbA_path) + " | sed 's/MEX/CYS/g; s/HID/HIS/g; s/HIE/HIS/g; s/HIP/HIS/g; s/MSE/MET/g; s/ASX/ASN/g; s/GLX/GLN/g; s/TYS/TRP/g' > " + target1_pdb, shell=True, check=True)
    target1_res_idx_type = extract_seq_and_dist_map(target1_pdb, target1_seq, target1_monomer_dist)
    if mode == "hetero":
        subprocess.run("grep \"^ATOM\" " + str(args.pdbB_path) + " | sed 's/MEX/CYS/g; s/HID/HIS/g; s/HIE/HIS/g; s/HIP/HIS/g; s/MSE/MET/g; s/ASX/ASN/g; s/GLX/GLN/g; s/TYS/TRP/g' > " + target2_pdb, shell=True, check=True)
        target2_res_idx_type = extract_seq_and_dist_map(target2_pdb, target2_seq, target2_monomer_dist)
    else:
        target2_res_idx_type = target1_res_idx_type

    ##### Step 1: Search MSA #####
    print("========== Step 1: Searching MSA ==========")
    if not os.path.isfile(target1_msa):
        print("Searching MSA for", target1_seq)
        subprocess.run(hhblits + " -i " + target1_seq + " -d " + UniRef_database + " -cpu " + str(args.n_cpu) + " -oa3m " + target1_msa + " -n 3 -e 0.001 -id 99 -cov 0.4", shell=True, check=True)
    if mode == "hetero" and not os.path.isfile(target2_msa):
        print("Searching MSA for", target2_seq)
        subprocess.run(hhblits + " -i " + target2_seq + " -d " + UniRef_database + " -cpu " + str(args.n_cpu) + " -oa3m " + target2_msa + " -n 3 -e 0.001 -id 99 -cov 0.4", shell=True, check=True)

    ##### Step 2: Extract Monomer MSA features
    print("========== Step 2: Extract Monomer MSA features ==========")
    if os.path.isfile(target1_msa):
        extract_MSA_features(target1, target1_msa, target1_hhm, target1_aln, target1_dca_di, target1_dca_apc, target1_esm_msa)

    if mode == "hetero" and os.path.isfile(target2_msa):
        extract_MSA_features(target2, target2_msa, target2_hhm, target2_aln, target2_dca_di, target2_dca_apc, target2_esm_msa)

    ##### Step 3: Extract paired MSA features
    print("========== Step 3: Extract paired MSA features ==========")
    if mode == "hetero":
        pairing_msa(target1_msa, target2_msa, paired_msa)
        extract_MSA_features(target, paired_msa, paired_hhm, paired_aln, paired_dca_di, paired_dca_apc, paired_esm_msa)

    ##### Step 4: Generate PPLM inter-protein attention matrix
    print("========== Step 4: Generate PPLM features ==========")
    if not os.path.isfile(pplm_feat):
        get_pplm_features(target1_seq, target2_seq, pplm_feat, device='cpu')

    ##### Step 5: Collect all features
    print("========== Step 5: Collect all features ==========")
    feats = collect_all_features()

    ##### Step 6: Predict inter-protein contact
    print("========== Step 6: Predict inter-protein contact ==========")
    if not os.path.isfile(pred_contact_pkl_path) or not os.path.isfile(pred_contact_txt_path):
        pred_inter_contatct = predict_contact(feats, mode, device=device)

        with open(pred_contact_pkl_path, "wb") as fw:
            pickle.dump(pred_inter_contatct, fw)

        with open(pred_contact_txt_path, "w") as fw:
            prediction_idx_prob = []
            for i in range(pred_inter_contatct.shape[0]):
                for j in range(pred_inter_contatct.shape[1]):
                    prediction_idx_prob.append([i, j, pred_inter_contatct[i, j]])

            prediction_idx_prob = sorted(prediction_idx_prob, key=lambda x: x[2], reverse=True)

            data = "{:<10}".format("Rank") + "{:<10}".format("ResIdx1") + "{:<10}".format("ResType1") + "{:<10}".format("ResIdx2") + "{:<10}".format("ResType2") + "{:<20}".format("Contact_Probability") + "\n"
            fw.write(data)
            for k in range(len(prediction_idx_prob)):
                res1 = prediction_idx_prob[k][0]
                res2 = prediction_idx_prob[k][1]
                prob = prediction_idx_prob[k][2]
                res1_idx = target1_res_idx_type[res1][0]
                res1_type = target1_res_idx_type[res1][1]
                res2_idx = target2_res_idx_type[res2][0]
                res2_type = target2_res_idx_type[res2][1]

                data = "{:<10}".format(k+1) + "{:<10}".format(str(res1_idx) + ":A") + "{:<10}".format(res1_type) + "{:<10}".format(str(res2_idx) + ":B") + "{:<10}".format(res2_type) + "{:<10}".format(f"{prob:.6g}") + "\n"
                fw.write(data)

def define_param(args):
    # param = {}
    global target1, target2, target, workspace, mode, device
    global target1_pdb, target1_seq, target1_monomer_dist, target2_pdb, target2_seq
    global target2_monomer_dist, target1_msa, target2_msa, target1_hhm, target1_aln
    global target1_dca_di, target1_dca_apc, target1_esm_msa, target2_hhm, target2_aln
    global target2_dca_di, target2_dca_apc, target2_esm_msa, paired_msa, paired_hhm
    global paired_aln, paired_dca_di, paired_dca_apc, paired_esm_msa, pplm_feat
    global pred_contact_pkl_path, pred_contact_txt_path


    target1 = args.pdbA_path.stem
    target2 = args.pdbB_path.stem
    target = str(args.output_folder).split('/')[-1]  # target1 + '-' + target2
    workspace = args.output_folder
    if not os.path.isdir(workspace):
        os.makedirs(workspace)

    assigned_device = "cuda:" + str(args.gpu_id)
    device = assigned_device if torch.cuda.is_available() else "cpu"

    if args.pdbA_path == args.pdbB_path:
        mode = "homo"
    else:
        mode = "hetero"

    print("device:", device, "   mode:", mode)

    target1_pdb = os.path.join(workspace, target1 + ".clean.pdb")
    target1_seq = os.path.join(workspace, target1 + ".fasta")
    target1_monomer_dist = os.path.join(workspace, target1 + ".monomer_dist.pkl")
    target2_pdb = os.path.join(workspace, target2 + ".clean.pdb")
    target2_seq = os.path.join(workspace, target2 + ".fasta")
    target2_monomer_dist = os.path.join(workspace, target2 + ".monomer_dist.pkl")
    target1_msa = os.path.join(workspace, target1 + ".a3m")
    target2_msa = os.path.join(workspace, target2 + ".a3m")
    target1_hhm = os.path.join(workspace, target1 + ".hhm")
    target1_aln = os.path.join(workspace, target1 + ".aln")
    target1_dca_di = os.path.join(workspace, target1 + ".dca_di")
    target1_dca_apc = os.path.join(workspace, target1 + ".dca_apc")
    target1_esm_msa = os.path.join(workspace, target1 + ".esm_msa.pkl")
    target2_hhm = os.path.join(workspace, target2 + ".hhm")
    target2_aln = os.path.join(workspace, target2 + ".aln")
    target2_dca_di = os.path.join(workspace, target2 + ".dca_di")
    target2_dca_apc = os.path.join(workspace, target2 + ".dca_apc")
    target2_esm_msa = os.path.join(workspace, target2 + ".esm_msa.pkl")
    paired_msa = os.path.join(workspace, target + "_paired.a3m")
    paired_hhm = os.path.join(workspace, target + "_paired.hhm")
    paired_aln = os.path.join(workspace, target + "_paired.aln")
    paired_dca_di = os.path.join(workspace, target + "_paired.dca_di")
    paired_dca_apc = os.path.join(workspace, target + "_paired.dca_apc")
    paired_esm_msa = os.path.join(workspace, target + "_paired.esm_msa.pkl")
    pplm_feat = os.path.join(workspace, target + ".pplm.pkl")
    pred_contact_pkl_path = os.path.join(workspace, target + ".pred_contact.pkl")
    pred_contact_txt_path = os.path.join(workspace, target + ".pred_contact.txt")

def extract_MSA_features(name, msa, hhm, aln, dci_di, dci_apc, esm_msa_path):

    print("Generating the HHM file for", msa)
    subprocess.run(hhmake + " -i " + msa + " -o " + hhm, shell=True, check=True)
    subprocess.run(reformat + " " + msa + " " + os.path.join(workspace, name + ".fas") + " -r -l 2000 >/dev/null", shell=True, check=True)
    subprocess.run("awk '{if(!($0~/^>/)){print}}' " + os.path.join(workspace, name + ".fas") + " > " + aln, shell=True, check=True)

    # generate the DCA features
    print("Generating the DCA features for", aln)
    if not os.path.isfile(dci_di) or not os.path.isfile(dci_apc):
        subprocess.run(ccmpred + " " + aln + " " + dci_di + " -R -A", shell=True, check=True)
        subprocess.run(ccmpred + " " + aln + " " + dci_apc + " -R", shell=True, check=True)

    # generate the ESM-MSA features
    print("Generating the ESM-MSA features for", msa)
    if not os.path.isfile(esm_msa_path):
        subprocess.run(hhfilter + " -i " + msa + " -o " + msa + "_filtered -diff 512", shell=True, check=True)
        subprocess.run(["python", esm_msa, esm_msa_model, msa + "_filtered", esm_msa_path])

def get_pplm_features(seqA_path, seqB_path, out_pkl_path, device='cpu'):
    mian_path = os.path.dirname(__file__)
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
        seqA, seqB = '', ''
        for line in open(seqA_path, "r").readlines():
            if not line.startswith(">"):
                seqA += line.strip()
        for line in open(seqB_path, "r").readlines():
            if not line.startswith(">"):
                seqB += line.strip()

        seqA_labels, seqA_strs, seqA_tokens = batch_converter([('seqA', seqA)])
        seqB_labels, seqB_strs, seqB_tokens = batch_converter([('seqB', seqB)])
        tokens = torch.cat([seqA_tokens, seqB_tokens], dim=-1).to(device)

        inter_chain_mask = torch.ones((len(seqA) + 2 + len(seqB) + 2, len(seqA) + 2 + len(seqB) + 2), device=device)
        inter_chain_mask[:len(seqA) + 2, :len(seqA) + 2] = 0
        inter_chain_mask[len(seqA) + 2:, len(seqA) + 2:] = 0

        ##### running PPLM #####
        out = model(tokens, inter_chain_mask, repr_layers=[33], need_head_weights=True, return_contacts=False)

        # attn_AA = out['attentions'].squeeze()[:, :, 1:(len(seqA) + 1), 1:(len(seqA) + 1)].reshape(33 * 20, len(seqA), len(seqA)).cpu().numpy()
        attn_AB = out['attentions'].squeeze()[:, :, 1:(len(seqA) + 1), -(len(seqB) + 1):-1].reshape(33 * 20, len(seqA), len(seqB)).cpu().numpy()
        attn_BA = out['attentions'].squeeze()[:, :, -(len(seqB) + 1):-1, 1:(len(seqA) + 1)].reshape(33 * 20, len(seqB), len(seqA)).cpu().numpy()
        # attn_BB = out['attentions'].squeeze()[:, :, -(len(seqB) + 1):-1, -(len(seqB) + 1):-1].reshape(33 * 20, len(seqB), len(seqB)).cpu().numpy()
        inter_attn = (attn_AB + attn_BA.transpose(0, 2, 1)) / 2

        with open(out_pkl_path, mode='wb') as fw:
            pickle.dump(inter_attn, fw)

def collect_all_features():

    with open(target1_monomer_dist, "rb") as fr:
        target1_M_dist = pickle.load(fr)

    target1_DCA_DI = np.expand_dims(np.loadtxt(target1_dca_di), 0)
    target1_DCA_APC = np.expand_dims(np.loadtxt(target1_dca_apc), 0)
    target1_PSSM = load_hmm(target1_hhm)['PSSM']

    with open(target1_esm_msa, "rb") as fr:
        esm_msa_data = pickle.load(fr)
        target1_esm_msa_1d = esm_msa_data['esm_msa_1d']
        target1_esm_msa_2d = esm_msa_data['row_attentions']

    print(target1_DCA_DI.shape, target1_DCA_APC.shape, target1_esm_msa_2d.shape, RBF(target1_M_dist).shape, target1_M_dist.shape)
    intra1_1d = np.concatenate([target1_PSSM, target1_esm_msa_1d], axis=-1).transpose(1,0)
    intra1_2d = np.concatenate([target1_DCA_DI, target1_DCA_APC, target1_esm_msa_2d, RBF(target1_M_dist)], axis=0)
    intra1_Mdist = target1_M_dist

    with open(pplm_feat, "rb") as fr:
        inter_pplm_attn = pickle.load(fr)

    if mode == "homo":
        intra2_1d = intra1_1d
        intra2_2d = intra1_2d
        intra2_Mdist = intra1_Mdist
        inter_2d = np.concatenate([target1_DCA_DI, target1_DCA_APC, target1_esm_msa_2d, inter_pplm_attn], axis=0)

    else:
        with open(target2_monomer_dist, "rb") as fr:
            target2_M_dist = pickle.load(fr)

        target2_DCA_DI = np.expand_dims(np.loadtxt(target2_dca_di), 0)
        target2_DCA_APC = np.expand_dims(np.loadtxt(target2_dca_apc), 0)
        target2_PSSM = load_hmm(target2_hhm)['PSSM']

        with open(target2_esm_msa, "rb") as fr:
            esm_msa_data = pickle.load(fr)
            target2_esm_msa_1d = esm_msa_data['esm_msa_1d']
            target2_esm_msa_2d = esm_msa_data['row_attentions']

        print(target2_DCA_DI.shape, target2_DCA_APC.shape, target2_esm_msa_2d.shape, RBF(target2_M_dist).shape, target2_M_dist.shape)
        intra2_1d = np.concatenate([target2_PSSM, target2_esm_msa_1d], axis=-1).transpose(1, 0)
        intra2_2d = np.concatenate([target2_DCA_DI, target2_DCA_APC, target2_esm_msa_2d, RBF(target2_M_dist)], axis=0)
        intra2_Mdist = target2_M_dist

        inter_DCA_DI = np.expand_dims(np.loadtxt(paired_dca_di), 0)
        inter_DCA_APC = np.expand_dims(np.loadtxt(paired_dca_apc), 0)
        with open(paired_esm_msa, "rb") as fr:
            esm_msa_data = pickle.load(fr)
            inter_esm_msa_2d = esm_msa_data['row_attentions']

        len1 = inter_pplm_attn.shape[-2]
        len2 = inter_esm_msa_2d.shape[-1]

        print(inter_DCA_DI.shape, inter_DCA_APC.shape, inter_esm_msa_2d.shape, inter_pplm_attn.shape)
        inter_2d = np.concatenate([inter_DCA_DI[:, :len1, len1:len1+len2], inter_DCA_APC[:, :len1, len1:len1+len2], inter_esm_msa_2d[:, :len1, len1:len1+len2], inter_pplm_attn], axis=0)

    feats = {"intra1_1d": intra1_1d, "intra1_2d": intra1_2d, "intra1_Mdist": intra1_Mdist, "intra2_1d": intra2_1d, "intra2_2d": intra2_2d, "intra2_Mdist": intra2_Mdist, "inter_2d": inter_2d}

    return feats

def predict_contact(feats, mode, device="cpu"):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if mode == "homo":
        model_paths = [os.path.join(script_dir, "pplm_contact/models/pplm_contact.homo_" + str(i) + ".pkl") for i in range(1, 6)]
    else:
        model_paths = [os.path.join(script_dir, "pplm_contact/models/pplm_contact.hetero_" + str(i) + ".pkl") for i in range(1, 6)]

    model = PPLM_Contact()
    model.to(device)

    intra1_1d = torch.Tensor(feats["intra1_1d"]).to(device).type(torch.float32).unsqueeze(0)
    intra1_2d = torch.Tensor(feats["intra1_2d"]).to(device).type(torch.float32).unsqueeze(0)
    intra1_Mdist = torch.Tensor(feats["intra1_Mdist"]).to(device).type(torch.float32).unsqueeze(0)
    intra2_1d = torch.Tensor(feats["intra2_1d"]).to(device).type(torch.float32).unsqueeze(0)
    intra2_2d = torch.Tensor(feats["intra2_2d"]).to(device).type(torch.float32).unsqueeze(0)
    intra2_Mdist = torch.Tensor(feats["intra2_Mdist"]).to(device).type(torch.float32).unsqueeze(0)
    inter_2d = torch.Tensor(feats["inter_2d"]).to(device).type(torch.float32).unsqueeze(0)

    ensemble_pred_inter_contact = []
    with torch.no_grad():
        for model_path in model_paths:

            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            #################################### Network predict #######################################
            contact_pred = model(intra1_1d, intra1_2d, intra2_1d, intra2_2d, inter_2d, intra1_Mdist, intra2_Mdist)

            pred_inter_contact = contact_pred  # [inter_contact_mask_ur]
            ensemble_pred_inter_contact.append(pred_inter_contact)

        pred_inter_contact = torch.stack(ensemble_pred_inter_contact)
        pred_inter_contact = torch.mean(pred_inter_contact, dim=0).squeeze().cpu().detach().numpy()

    return pred_inter_contact





if __name__ == "__main__":
    main()