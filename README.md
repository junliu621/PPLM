
# PPLM: A Protein-Protein Language Model for Interaction, Binding Affinity, and Interface Contact Prediction

![PPLM Banner](https://zhanglab.comp.nus.edu.sg/PPLM/img/pplm_pipeline.jpg)

---

**Version 1.0, 03/25/2025**  
(Copyrighted by the Regents of the National University of Singapore, All rights reserved)

PPLM is a protein–protein language model that learns directly from paired sequences through a novel
attention architecture, explicitly capturing inter-protein context. Building on PPLM, we developed
PPLM-PPI, PPLM-Affinity, and PPLM-Contact for predicting protein–protein interactions, estimating
binding affinity, and identifying interface residue contacts, respectively.

**Authors**: Jun Liu, Hungyu Chen, and Yang Zhang

**Contact**: junl_sg@nus.edu.sg

**Citation**:  
> Jun Liu, Hungyu Chen, Yang Zhang. A Protein-Protein Language Model for Interaction, Binding Affinity, and Interface Contact Prediction. *In preparation.*

**Webserver**: [PPLM Online Submission](https://zhanggroup.org/PPLM/)  

**License**: PolyForm Noncommercial License

---

## System Requirements
- x86_64 machine
- Linux Kernel OS

## Software & Dataset Requirements (for PPLM-Contact)
1. **HH-suite3** for MSA Search: [Install HH-suite3](https://github.com/soedinglab/hh-suite)
2. **Uniclust Database**: [Download Uniclust](http://wwwuser.gwdg.de/~compbiol/uniclust/2021_03/)
3. **CCMpred** for DCA: [Install CCMpred](https://github.com/soedinglab/CCMpred)
4. **LoadHHM** for PSSM Calculation: [LoadHHM.py](https://github.com/j3xugit/RaptorX-Contact/blob/master/Common/LoadHHM.py)
5. **ESM-MSA** for Feature Generation: [Install ESM](https://github.com/facebookresearch/esm)

## Download parameters
- Run the download_parameter.sh script located in the models/ folder of pplm/, pplm_ppi/, pplm_affinity/, and pplm_contact/

---

## Usage

### 1. Install environment
```bash
conda env create -f environment.yml
```
### 2. Activate environment
```bash
conda activate PPLM
```
### 3. Run PPLM-PPI
```bash
python run_pplm-ppi.py example/seq_pairs.fasta example/seq_pairs.results
```
### 4. Run PPLM-Affinity
```bash
python run_pplm-affinity.py example/receptor.fasta example/ligand.fasta
```
### 5. Run PPLM-Contact (homodimer)
```bash
python run_pplm-contact.py example/protein.pdb example/protein.pdb example/homo_example
```
### 6. Run PPLM-Contact (heterodimer)
```bash
python run_pplm-contact.py example/protein1.pdb example/protein2.pdb example/hetero_example
```
### 7. Generate embeddings and attention weights for other applications
```bash
python run_pplm.py example/seq1.fasta example/seq2.fasta example/seq1-seq2.pplm.pkl
```

## Example Outputs

### PPLM-PPI
- Command:
```bash
python run_pplm-ppi.py example/seq_pairs.fasta example/seq_pairs.results
```
- Output: The predicted interaction probabilities are saved in example/seq_pairs.results:
```
>10090.ENSMUSP00000085394:10090.ENSMUSP00000116785
0.001926
>10090.ENSMUSP00000043111:10090.ENSMUSP00000102211
0.991765
>10090.ENSMUSP00000134644:10090.ENSMUSP00000131939
0.000425
>10090.ENSMUSP00000104648:10090.ENSMUSP00000095136
0.060997
>10090.ENSMUSP00000131855:10090.ENSMUSP00000118766
0.004577
>10090.ENSMUSP00000008036:10090.ENSMUSP00000046016
0.929329
...

Each entry consists of:
• Protein Pair: Represented in the format >Protein1:Protein2.
• Interaction Probability: The likelihood of interaction between the given protein pair.
```

### PPLM-Affinity
- Command:
```bash
python run_pplm-affinity.py example/receptor.fasta example/ligand.fasta
```
- Output: Predicted binding affinity printed to the command line：
```
Predicted binding affinity: -7.6090136
```

### PPLM-Contact
- Command:
```bash
python run_pplm-contact.py example/protein.pdb example/protein.pdb example/homo_example
```
- Output: The predicted contacts are saved in example/homo_example/homo_example.pred_contact.txt:
```
Format:
Rank      ResIdx1   ResType1  ResIdx2   ResType2  Contact_Probability
1         23:A      MET       26:B      CYS       0.976151
2         26:A      CYS       23:B      MET       0.974481
3         22:A      ILE       26:B      CYS       0.971633
4         23:A      MET       30:B      GLN       0.971191
5         30:A      GLN       22:B      ILE       0.970514
6         27:A      GLY       23:B      MET       0.970334
7         22:A      ILE       30:B      GLN       0.970124
8         30:A      GLN       23:B      MET       0.96919
9         23:A      MET       27:B      GLY       0.966725
10        23:A      MET       23:B      MET       0.966512
...
```

