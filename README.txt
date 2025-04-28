# PPLM: Protein-Protein Language Model

![PPLM Logo](https://zhanglab.comp.nus.edu.sg/PPLM/logo.png)

**Version 1.0 (03/25/2025)**

(Copyrighted by the Regents of the National University of Singapore, All rights reserved)

PPLM is a protein–protein language model that learns directly from paired sequences through a novel attention architecture, explicitly capturing inter-protein context. Building on PPLM, we developed:

- **PPLM-PPI**: Predicts protein–protein interactions
- **PPLM-Affinity**: Estimates binding affinity
- **PPLM-Contact**: Identifies interface residue contacts

**Authors**: Jun Liu, Hungyu Chen, and Yang Zhang  
**Contact**: junl_sg@nus.edu.sg  
**Website**: [PPLM Online Server](https://zhanglab.comp.nus.edu.sg/PPLM/)

If you use this program, please cite:
> Jun Liu, Hungyu Chen, Yang Zhang. A Protein-Protein Language Model for Interaction, Binding Affinity, and Interface Contact Prediction. *In preparation.*

---

## System Requirements
- x86_64 machine
- Linux kernel OS

## Software & Dataset Requirements for PPLM-Contact

1. **HH-suite3** for MSA Search
   - Install from [HH-suite GitHub](https://github.com/soedinglab/hh-suite)
   - Set `hhsuite_dir` in `pplm_contact/config.py`

2. **Uniclust Database**
   - Download from [Uniclust 2021_03](http://wwwuser.gwdg.de/~compbiol/uniclust/2021_03/)
   - Set `UniRef_database` in `config.py`

3. **CCMpred** for Direct Coupling Analysis (DCA)
   - Install from [CCMpred GitHub](https://github.com/soedinglab/CCMpred)
   - Or use the pre-packaged version in `pplm_contact/external_tools`
   - Set `ccmpred` path in `config.py`

4. **LoadHHM** for PSSM Calculation
   - Download [LoadHHM.py](https://github.com/j3xugit/RaptorX-Contact/blob/master/Common/LoadHHM.py)
   - Place in `pplm_contact` directory or use the provided version

5. **ESM-MSA** for Feature Generation
   - Install from [ESM GitHub](https://github.com/facebookresearch/esm)
   - Pre-trained model: [esm_msa1_t12_100M_UR50S.pt](https://dl.fbaipublicfiles.com/fair-esm/models/esm_msa1_t12_100M_UR50S.pt)
   - Set `esm_msa_model` in `config.py`

---

## Installation

1. Install PPLM environment:

```bash
conda env create -f environment.yml
```

2. Activate PPLM environment:

```bash
conda activate PPLM
```

---

## Usage

### 1. Run PPLM-PPI for Paired Sequences
```bash
python run_pplm-ppi.py example/seq_pairs.fasta example/seq_pairs.results
```

### 2. Run PPLM-Affinity for Receptor and Ligand
```bash
python run_pplm-contact.py example/receptor.fasta example/ligand.fasta
```

### 3. Run PPLM-Contact for Homodimer
```bash
python run_pplm-contact.py example/protein.pdb example/protein.pdb example/homo_example
```

### 4. Run PPLM-Contact for Heterodimer
```bash
python run_pplm-contact.py example/protein1.pdb example/protein2.pdb example/hetero_example
```

### 5. Run PPLM-PPI for Individual Sequences
```bash
python pplm_ppi/predict.py example/seq1.fasta example/seq2.fasta
```

### 6. Generate Embeddings and Attention Weights
```bash
python run_pplm.py example/seq1.fasta example/seq2.fasta example/seq1-seq2.pplm.pkl
```

---

## Example Outputs

### PPLM-PPI
- Predicted interaction probabilities saved in `.results` file:
```
>Protein1:Protein2
Interaction Probability
```

Example:
```
>10090.ENSMUSP00000085394:10090.ENSMUSP00000116785
0.001926
>10090.ENSMUSP00000043111:10090.ENSMUSP00000102211
0.991765
...
```

### PPLM-Affinity
- Predicted binding affinity printed to console.

Example output:
```
Predicted binding affinity: -7.6090136
```

### PPLM-Contact
- Predicted contacts saved in `.pred_contact.txt` file:

| Rank | ResIdx1 | ResType1 | ResIdx2 | ResType2 | Contact_Probability |
|:----:|:-------:|:--------:|:-------:|:--------:|:-------------------:|
| 1    | 23:A    | MET      | 26:B    | CYS      | 0.976151             |
| 2    | 26:A    | CYS      | 23:B    | MET      | 0.974481             |
| ...  | ...     | ...      | ...     | ...      | ...                 |

---

## License

The source code is freely available to academic and non-profit users under the PolyForm Noncommercial License.

---

## Bug Reports

For any issues or inquiries, please contact:

**Jun Liu**  
Email: junl_sg@nus.edu.sg

