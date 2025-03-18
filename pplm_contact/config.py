import os

hhsuite_dir = "/mnt/rna01/junl/tools/hh-suite/build"

hhblits = os.path.join(hhsuite_dir, "bin/hhblits")
hhmake = os.path.join(hhsuite_dir, "bin/hhmake")
reformat = os.path.join(hhsuite_dir, "scripts/reformat.pl")
hhfilter = os.path.join(hhsuite_dir, "bin/hhfilter")
UniRef_database = "/mnt/dna01/library2/e2e_folding/alphafold/v2.3/lib/uniref30/UniRef30_2021_03"


ccmpred = "/mnt/rna01/junl/PPLM/PPLM_source_code/pplm_contact/external_tools/ccmpred"
esm_msa = "/mnt/rna01/junl/PPLM/PPLM_source_code/pplm_contact/external_tools/extract_esm_msa_features.py"
esm_msa_model = "/mnt/rna01/junl/tools/esm-main/esm_models/esm_msa1_t12_100M_UR50S.pt"


