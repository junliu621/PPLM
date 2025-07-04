import numpy as np
import pickle
import string

restype_3to1 = {k: v for k, v in zip(['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'], 'ARNDCEQGHILKMFPSTWYV')}

heavy_atoms = ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG']

def pdb2seq(pdb_path, seq_path):
    sequence_list = []
    sequence = ""
    last_chain_id = ""
    for data in open(pdb_path, 'r').readlines():
        atom_name = data[13:16].strip()
        if data.startswith("ATOM") and atom_name == 'CA':
            chain_id = data[21]
            res_name = data[17:20].strip()

            if chain_id != last_chain_id and len(sequence) > 0:
                sequence_list.append([last_chain_id, sequence])
                sequence = restype_3to1[res_name]
            else:
                sequence += restype_3to1[res_name]

            last_chain_id = chain_id

    if len(sequence) > 0:
        sequence_list.append([last_chain_id, sequence])

    if len(sequence_list) > 1:
        print("Warning:", pdb_path, "contain multiple chains:", sequence_list[:, 0], "! Only first chain is considered.")

    return sequence_list[0][1]

    with open(seq_path, 'w') as fw:
        chain_id = sequence_list[0][0]
        sequence = sequence_list[0][1]
        fw.write(">seq_" + chain_id + " " + str(len(sequence)) + "\n")
        fw.write(sequence + "\n")

def extract_seq_and_dist_map(pdb_path, seq_path, dist_path):
    ### Load pdb_chain ###
    pdb_res_coordis = []
    res_atom_coordis = {}
    res_idx_type = []
    sequence = ''
    last_res_idx = -1
    last_res_name = ''
    for data in open(pdb_path, 'r').readlines():
        if data.startswith("ATOM"):
            atom_name = data[13:16].strip()
            res_name = data[17:20].strip()
            res_idx = int(data[22:26].strip())
            coordi_x = float(data[30:38].strip())
            coordi_y = float(data[38:46].strip())
            coordi_z = float(data[46:54].strip())
            if last_res_idx != -1 and res_idx != last_res_idx:
                pdb_res_coordis.append(res_atom_coordis)
                res_atom_coordis = {}
                sequence += restype_3to1[last_res_name]
                res_idx_type.append([last_res_idx, last_res_name])
            res_atom_coordis[atom_name] = [coordi_x, coordi_y, coordi_z]
            last_res_idx = res_idx
            last_res_name = res_name
    if len(res_atom_coordis) != 0:
        pdb_res_coordis.append(res_atom_coordis)
        res_atom_coordis = {}
        sequence += restype_3to1[last_res_name]
        res_idx_type.append([last_res_idx, last_res_name])

    length = len(sequence)

    ############## extract heavy atom distance ##################
    heavy_atom_dist_map = np.ones((1, length, length)) * np.inf
    for i in range(length):
        for j in range(i, length):
            min_dist = np.inf
            for heay_i in heavy_atoms:
                if heay_i in pdb_res_coordis[i]:
                    coordi_1 = pdb_res_coordis[i][heay_i]
                else:
                    continue
                for heay_j in heavy_atoms:
                    if heay_j in pdb_res_coordis[j]:
                        coordi_2 = pdb_res_coordis[j][heay_j]
                    else:
                        continue
                    dist = np.sqrt(pow(coordi_1[0] - coordi_2[0], 2) + pow(coordi_1[1] - coordi_2[1], 2) + pow(coordi_1[2] - coordi_2[2], 2))
                    if dist < min_dist:
                        min_dist = dist
            heavy_atom_dist_map[0, i, j] = min_dist
            heavy_atom_dist_map[0, j, i] = min_dist

    with open(dist_path, mode='wb') as fw:
        pickle.dump(heavy_atom_dist_map, fw)

    with open(seq_path, 'w') as fw:
        fw.write(">seq " + str(length) + "\n")
        fw.write(sequence + "\n")

    return res_idx_type

def extract_seq_and_dist_map_dimer(pdb_path, seqA_path, seqB_path, chain_A_dist_path, chain_B_dist_path, inter_chain_dist_path):
    ### Load pdb_chain ###
    chiains_data = {'pdb_res_coordis': [], 'res_idx_type': [], 'sequence': []}
    pdb_res_coordis = []
    res_atom_coordis = {}
    res_idx_type = []
    sequence = ''
    last_res_idx = -1
    last_res_name = ''
    last_chain_id = ''
    for data in open(pdb_path, 'r').readlines():
        if data.startswith("ATOM"):
            atom_name = data[13:16].strip()
            res_name = data[17:20].strip()
            chain_id = data[21].strip()
            res_idx = int(data[22:26].strip())
            coordi_x = float(data[30:38].strip())
            coordi_y = float(data[38:46].strip())
            coordi_z = float(data[46:54].strip())

            if last_chain_id != '' and last_chain_id != chain_id:
                if last_res_idx != -1 and res_idx != last_res_idx:
                    pdb_res_coordis.append(res_atom_coordis)
                    res_atom_coordis = {}
                    sequence += restype_3to1[last_res_name]
                    res_idx_type.append([last_res_idx, last_res_name])

                chiains_data['pdb_res_coordis'].append(pdb_res_coordis)
                chiains_data['res_idx_type'].append(res_idx_type)
                chiains_data['sequence'].append(sequence)

                pdb_res_coordis = []
                res_atom_coordis = {}
                res_idx_type = []
                sequence = ''
                last_res_idx = -1
                last_res_name = ''

            if last_res_idx != -1 and res_idx != last_res_idx:
                pdb_res_coordis.append(res_atom_coordis)
                res_atom_coordis = {}
                sequence += restype_3to1[last_res_name]
                res_idx_type.append([last_res_idx, last_res_name])
            res_atom_coordis[atom_name] = [coordi_x, coordi_y, coordi_z]
            last_res_idx = res_idx
            last_res_name = res_name
            last_chain_id = chain_id

    if len(res_atom_coordis) != 0:
        pdb_res_coordis.append(res_atom_coordis)
        res_atom_coordis = {}
        sequence += restype_3to1[last_res_name]
        res_idx_type.append([last_res_idx, last_res_name])

        chiains_data['pdb_res_coordis'].append(pdb_res_coordis)
        chiains_data['res_idx_type'].append(res_idx_type)
        chiains_data['sequence'].append(sequence)

    # print("chiains_data:", len(chiains_data['pdb_res_coordis']), len(chiains_data['res_idx_type']), len(chiains_data['sequence']))

    if len(chiains_data['pdb_res_coordis']) < 2:
        print("Error:", pdb_path, "has less than 2 chains!!!")
        exit()
    elif len(chiains_data['pdb_res_coordis']) > 2:
        print("Warning:", pdb_path, "has more than 2 chains, only the first two are considered!!!")


    ################## Get the coordinates, residue type, and sequence of the first two chians ##################
    pdbA_res_coordis = chiains_data['pdb_res_coordis'][0]
    pdbA_res_idx_type = chiains_data['res_idx_type'][0]
    pdbA_sequence = chiains_data['sequence'][0]

    pdbB_res_coordis = chiains_data['pdb_res_coordis'][1]
    pdbB_res_idx_type = chiains_data['res_idx_type'][1]
    pdbB_sequence = chiains_data['sequence'][1]

    ############## extract distance map of chain A ##################
    len_A = len(pdbA_res_coordis)
    chainA_dist_map = np.ones((1, len_A, len_A)) * np.inf
    for i in range(len_A):
        for j in range(i, len_A):
            min_dist = np.inf
            for heay_i in heavy_atoms:
                if heay_i in pdbA_res_coordis[i]:
                    coordi_1 = pdbA_res_coordis[i][heay_i]
                else:
                    continue
                for heay_j in heavy_atoms:
                    if heay_j in pdbA_res_coordis[j]:
                        coordi_2 = pdbA_res_coordis[j][heay_j]
                    else:
                        continue
                    dist = np.sqrt(pow(coordi_1[0] - coordi_2[0], 2) + pow(coordi_1[1] - coordi_2[1], 2) + pow(coordi_1[2] - coordi_2[2], 2))
                    if dist < min_dist:
                        min_dist = dist
            chainA_dist_map[0, i, j] = min_dist
            chainA_dist_map[0, j, i] = min_dist

    with open(chain_A_dist_path, mode='wb') as fw:
        pickle.dump(chainA_dist_map, fw)

    with open(seqA_path, 'w') as fw:
        fw.write(">seqA " + str(len_A) + "\n")
        fw.write(pdbA_sequence + "\n")

    ############## extract distance map of chain B ##################
    len_B = len(pdbB_res_coordis)
    chainB_dist_map = np.ones((1, len_B, len_B)) * np.inf
    for i in range(len_B):
        for j in range(i, len_B):
            min_dist = np.inf
            for heay_i in heavy_atoms:
                if heay_i in pdbB_res_coordis[i]:
                    coordi_1 = pdbB_res_coordis[i][heay_i]
                else:
                    continue
                for heay_j in heavy_atoms:
                    if heay_j in pdbB_res_coordis[j]:
                        coordi_2 = pdbB_res_coordis[j][heay_j]
                    else:
                        continue
                    dist = np.sqrt(pow(coordi_1[0] - coordi_2[0], 2) + pow(coordi_1[1] - coordi_2[1], 2) + pow(coordi_1[2] - coordi_2[2], 2))
                    if dist < min_dist:
                        min_dist = dist
            chainB_dist_map[0, i, j] = min_dist
            chainB_dist_map[0, j, i] = min_dist

    with open(chain_B_dist_path, mode='wb') as fw:
        pickle.dump(chainB_dist_map, fw)

    with open(seqB_path, 'w') as fw:
        fw.write(">seqA " + str(len_B) + "\n")
        fw.write(pdbB_sequence + "\n")

    ############## extract inter-chain distance map of chain A and B ##################
    inter_chain_dist_map = np.ones((1, len_A, len_B)) * np.inf
    for i in range(len_A):
        for j in range(len_B):
            min_dist = np.inf
            for heay_i in heavy_atoms:
                if heay_i in pdbA_res_coordis[i]:
                    coordi_1 = pdbA_res_coordis[i][heay_i]
                else:
                    continue
                for heay_j in heavy_atoms:
                    if heay_j in pdbB_res_coordis[j]:
                        coordi_2 = pdbB_res_coordis[j][heay_j]
                    else:
                        continue
                    dist = np.sqrt(pow(coordi_1[0] - coordi_2[0], 2) + pow(coordi_1[1] - coordi_2[1], 2) + pow(coordi_1[2] - coordi_2[2], 2))
                    if dist < min_dist:
                        min_dist = dist
            inter_chain_dist_map[0, i, j] = min_dist

    with open(inter_chain_dist_path, mode='wb') as fw:
        pickle.dump(inter_chain_dist_map, fw)

    return pdbA_sequence, pdbA_res_idx_type, pdbB_sequence, pdbB_res_idx_type

def pairing_msa(msa1_path, msa2_path, paired_msa_path):
    msas1, sid1 = extract_taxid(msa1_path)
    msas2, sid2 = extract_taxid(msa2_path)
    aligns = alignment(msas1, sid1, msas2, sid2, top=True)

    with open(paired_msa_path, 'w') as f:
        f.write(">target " + str(len(aligns[0])) + "\n")
        f.write(aligns[0] + "\n")

        for idx, aligned_seq in enumerate(aligns[1:]):
            f.write(">seq" + str(idx+1) + "\n")
            f.write(aligned_seq + "\n")
def extract_taxid(file, gap_cutoff=0.8):
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)

    lines = open(file, 'r').readlines()
    query = lines[1].strip().translate(translation)
    seq_len = len(query)

    msas = [query]
    sid = [0]
    for line in lines[2:]:

        if line[0] == ">":
            if "TaxID=" in line:
                content = line.split("TaxID=")[1]
                if len(content) > 0:
                    try:
                        sid.append(int(content.split()[0]))
                    except:
                        sid.append(0)
            elif "OX=" in line:
                content = line.split("OX=")[1]
                if len(content) > 0:
                    try:
                        sid.append(int(content.split()[0]))
                    except:
                        sid.append(0)
            else:
                sid.append(0)
            continue

        seq = line.strip().translate(translation)
        gap_fra = float(seq.count('-')) / seq_len
        if gap_fra <= gap_cutoff:
            msas.append(seq)
        else:
            sid.pop(-1)

    if len(msas) != len(sid):
        print("ERROR: len(msas) != len(sid)")
        print(len(msas), len(sid))
        exit()

    return msas, np.array(sid)

def cal_identity(query, sub_msas):
    """
    Args:
        query : str
        sub_msas : List[str]
    Return:
        identity : np.array
    """

    identity = np.zeros((len(sub_msas)))
    seq_len = len(query)
    ones = np.ones(seq_len)
    for idx, seq in enumerate(sub_msas):
        match = [query[i] == seq[i] for i in range(seq_len)]
        counts = np.sum(ones[match])
        identity[idx] = counts / seq_len

    return identity

def alignment(msas1, sid1, msas2, sid2, top=True):
    # obtain the same species and delete species=0
    smatch = np.intersect1d(sid1, sid2)
    smatch = smatch[np.argsort(smatch)]
    smatch = np.delete(smatch, 0)

    query1 = msas1[0]
    query2 = msas2[0]
    aligns = [query1 + query2]

    for id in smatch:

        index1 = np.where(sid1 == id)[0]
        sub_msas1 = [msas1[idx] for idx in index1]
        identity1 = cal_identity(query1, sub_msas1)
        sort_idx1 = np.argsort(-identity1)

        index2 = np.where(sid2 == id)[0]
        sub_msas2 = [msas2[idx] for idx in index2]
        identity2 = cal_identity(query2, sub_msas2)
        sort_idx2 = np.argsort(-identity2)

        if top == True:
            aligns.append(sub_msas1[sort_idx1[0]] + \
                          sub_msas2[sort_idx2[0]])
        else:
            num = min(len(sub_msas1), len(sub_msas2))
            for i in range(num):
                aligns.append(sub_msas1[sort_idx1[i]] + \
                              sub_msas2[sort_idx2[i]])

    return aligns


def RBF(dist_map):
    # Radial Basis Function
    D_min, D_max, D_count = 2., 22., 64
    D_mu = np.linspace(D_min, D_max, D_count)
    D_mu = D_mu[None,:]
    D_sigma = (D_max - D_min) / D_count

    dist_map = dist_map.transpose(1,2,0)
    RBF = np.exp(-((dist_map - D_mu) / D_sigma)**2)

    return RBF.transpose(2,0,1)
