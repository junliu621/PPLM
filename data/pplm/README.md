############################### PPLM paired sequence datasets ###############################

1.  The sequence pairs in trainset.PDB.txt and validset.PDB.txt were collected from the PDB 
    database and clustered, and the pairs in the same line belong to the same cluster. Each 
    sequence pair is named using the format: $pdb_id"-assembly1_"$chain1_id"_"$chain2_id,
    for example, 5hrz-assembly1_B_C, where chains B and C form a sequence pair.

    The PDB_seqs folder stores the single-chain sequences of each pair. You can find the two 
    sequences of the pair $pdb_id"-assembly1_"$chain1_id"_"$chain2_id through the files         
    $pdb_id"-assembly1_"$chain1_id".fa and $pdb_id"-assembly1_"$chain2_id".fa. For example, 
    the sequence files of 5hrz-assembly1_B_C are 5hrz-assembly1_B.fa and 5hrz-assembly1_C.fa.

    You can also download the sequence files from the PDB website (https://www.rcsb.org/) 
    according to the corresponding pdb_id and chain_id.


2.  The sequence pairs in trainset.STRING.txt and validset.STRING.txt were collected fromthe 
    STRING database and clustered, and the pairs in the same line belong to the same cluster.
    Each sequence pair is named using the format: $seq1_id"-"$seq2_id, for example,
    1736365.ASG30_21700-1736365.ASG30_17300, which consists of sequences 1736365.ASG30_21700
    and 1736365.ASG30_17300.

    The STRING_seqs folder stores the sequences of each pair. You can find the two sequences
    of the pair $seq1_id"-"$seq2_id through $seq1_id.fa and $seq2_id.fa. For example, the 
    sequence files of 1736365.ASG30_21700-1736365.ASG30_17300 are 1736365.ASG30_21700.fa and 
    1736365.ASG30_17300.fa.

    You can also download the sequences from the STRING website (https://string-db.org/).
