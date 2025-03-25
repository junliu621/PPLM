# PPLM
Protein-Protein Language Model with Novel Attention Mechanisms for Enhanced Inter-Protein Contact and Interaction Prediction

####################################################################################################

                                  _____   _____   _       __  __
                                 |  __ \ |  __ \ | |     |  \/  |
                                 | |__) || |__) || |     | \  / |
                                 |  ___/ |  ___/ | |     | |\/| |
                                 | |     | |     | |____ | |  | |
                                 |_|     |_|     |______||_|  |_|


                                    (Version 1.0, 03/25/2025)

(Copyrighted by the Regents of the National University of Singapore, All rights reserved)

PPLM is a specialized protein-protein language model designed for predicting protein-protein
contacts and interactions.

Author: Jun Liu

For bug reports and inquiries, please contact: junl_sg@nus.edu.sg

If you use this program, please cite:
Jun Liu, Hungyu Chen, Yang Zhang. Protein-Protein Language Model with Novel Attention Mechanisms
for Enhanced Inter-Protein Contact and Interaction Prediction. In preparation.

This is the stand-alone program. Alternatively, users can submit jobs online at:
https://zhanglab.comp.nus.edu.sg/PPLM/

The source code is freely available to academic and non-profit users under the PolyForm
Noncommercial License

####################################################################################################


####################################### System Requirements ########################################
x86_64 machine, Linux kernel OS.

################################# Software & Dataset Requirements ##################################
1.  HH-suite3 for MSA Search.
    Install HH-suite3 (https://github.com/soedinglab/hh-suite) and set the "hhsuite_dir" parameter
    in the "pplm_contact/config.py" file.

2.  Uniclust Database for MSA Search.
    Download the Uniclust database (http://wwwuser.gwdg.de/~compbiol/uniclust/2021_03/)and set the
    "UniRef_database" parameter in the "config.py" file.

3.  CCMpred for Direct Coupling Analysis (DCA).
    Install ccmpred (https://github.com/soedinglab/CCMpred), or use the pre-packaged version in the
    "pplm_contact/external_tools" directory. Set the "ccmpred" parameter in the "config.py" file.

4.  LoadHHM for PSSM Calculation.
    Download "LoadHHM.py" (https://github.com/j3xugit/RaptorX-Contact/blob/master/Common/LoadHHM.py)
    and place the file in the "pplm_contact" directory of the PPLM package, or use the pre-packaged
    version within the "pplm_contact" directory.

5.  ESM-MSA for Feature Generation.
    Install the ESM package (https://github.com/facebookresearch/esm), or use the pre-packaged
    version within "pplm_contact/external_tools" directory.

    Download the pre-trained model and set the "esm_msa_model" parameter in the "config.py" file.
    (https://dl.fbaipublicfiles.com/fair-esm/models/esm_msa1_t12_100M_UR50S.pt)


############################################## Usage ###############################################
1.  Install PPLM environment:
	conda env create -f environment.yml

2.  Activate PPLM environment:
    conda activate PPLM

3.  Run PPLM-Contact for protein homodimer:
    python run_pplm-contact.py example/protein.pdb example/protein.pdb example/homo_example

4.  Run PPLM-Contact for protein heterodimer:
    python run_pplm-contact.py example/protein1.pdb example/protein2.pdb example/hetero_example

5.  Run PPLM-PPI for a batch of paired sequences:
    python run_pplm-ppi.py example/seq_pairs.fasta example/seq_pairs.results

    You can also run PPLM-PPI for two individual sequences:
    python pplm_ppi/predict.py example/seq1.fasta example/seq2.fasta

6.  Run PPLM to generate embeddings and attention weights for other applications:
    python run_pplm.py example/seq1.fasta example/seq2.fasta example/seq1-seq2.pplm.pkl

########################################## Example Output ##########################################
1.  Output of PPLM-Contact:
    (a) Run PPLM-Contact:
        python run_pplm-contact.py example/protein.pdb example/protein.pdb example/homo_example

    (b) The predicted contacts are saved in example/homo_example/homo_example.pred_contact.txt.
        The file is structured as follows:
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

        • ResIdx1 and ResIdx2: Residue indexes of first (A) and second (B) proteins, respectively.
        • ResTyep1 and ResType2: Amino acid types corresponding to ResIdx1 and ResIdx2.
        • Contact_Probability: The predicted probability of residue contact.

2.  Output of PPLM-PPI:
    (a) Run PPLM-PPI:
        python run_pplm-ppi.py example/seq_pairs.fasta example/seq_pairs.results

    (b) The predicted interaction probabilities are saved in example/seq_pairs.results.
        The file is structured as follows:
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

####################################################################################################
