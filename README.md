
# PPLM

![PPLM Banner](https://zhanglab.comp.nus.edu.sg/PPLM/img/pipeline.png)) <!-- 可以换成真实Banner -->

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

**Webserver**: [PPLM Online Submission](https://zhanglab.comp.nus.edu.sg/PPLM/)  

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

## Usage
```bash
# 1. Install environment
conda env create -f environment.yml

# 2. Activate environment
conda activate PPLM

# 3. Run PPLM-PPI
python run_pplm-ppi.py example/seq_pairs.fasta example/seq_pairs.results

# 4. Run PPLM-Affinity
python run_pplm-affinity.py example/receptor.fasta example/ligand.fasta

# 5. Run PPLM-Contact (homodimer)
python run_pplm-contact.py example/protein.pdb example/protein.pdb example/homo_example

# 6. Run PPLM-Contact (heterodimer)
python run_pplm-contact.py example/protein1.pdb example/protein2.pdb example/hetero_example

# 7. Generate embeddings and attention weights
python run_pplm.py example/seq1.fasta example/seq2.fasta example/seq1-seq2.pplm.pkl
```

## Example Outputs

### PPLM-PPI
- Command:
```bash
python run_pplm-ppi.py example/seq_pairs.fasta example/seq_pairs.results
```
- Output format (example):
```
>Protein1:Protein2
Interaction Probability
```

### PPLM-Affinity
- Command:
```bash
python run_pplm-affinity.py example/receptor.fasta example/ligand.fasta
```
- Output: Predicted binding affinity printed to the command line.

### PPLM-Contact
- Command:
```bash
python run_pplm-contact.py example/protein.pdb example/protein.pdb example/homo_example
```
- Output file: `homo_example/homo_example.pred_contact.txt`

Format:
| Rank | ResIdx1 | ResType1 | ResIdx2 | ResType2 | Contact Probability |
|:----:|:-------:|:--------:|:-------:|:--------:|:-------------------:|

---

*This README was automatically generated.*
