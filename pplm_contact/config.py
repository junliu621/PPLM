import os

hhsuite_dir = "/XXX/hh-suite/build"

hhblits = os.path.join(hhsuite_dir, "bin/hhblits")
hhmake = os.path.join(hhsuite_dir, "bin/hhmake")
reformat = os.path.join(hhsuite_dir, "scripts/reformat.pl")
hhfilter = os.path.join(hhsuite_dir, "bin/hhfilter")

UniRef_database = "/XXX/uniref30/UniRef30_2021_03"
esm_msa_model = "/XXX/esm_msa1_t12_100M_UR50S.pt"

pplm_contact_dir = os.path.dirname(os.path.abspath(__file__))
ccmpred = os.path.join(pplm_contact_dir, "external_tools/ccmpred")
esm_msa = os.path.join(pplm_contact_dir, "external_tools/extract_esm_msa_features.py")
