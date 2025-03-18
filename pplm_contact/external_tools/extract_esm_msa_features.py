import numpy as np
import esm
import torch
import argparse
import string
import itertools
import pickle
from Bio import SeqIO
from typing import List, Tuple

def main():
    parser = argparse.ArgumentParser(description="Extract ESM MSA features")

    parser.add_argument("model_path",
                        type=str,
                        help="Location of esm-msa parameter model")

    parser.add_argument("msa_path",
                        type=str,
                        help="Location of msa")

    parser.add_argument("feat_path",
                        type=str,
                        help="Location of output features pkl file")

    parser.add_argument("--device",
                        type=str,
                        default="cpu",
                        help="device specified",
                        )

    args = parser.parse_args()


    # load model and read msa
    print("model_path:", args.model_path)
    esm1b, esm1b_batch_converter = load_esm(args.model_path)
    msa_data = [read_msa(args.msa_path, 512)]
    # convert the sequence to tokens
    esm1b_batch_labels, esm1b_batch_strs, esm1b_batch_tokens = esm1b_batch_converter(msa_data)

    esm1b = esm1b.to(args.device)
    esm1b_batch_tokens = esm1b_batch_tokens.to(args.device)

    with torch.no_grad():
        results = esm1b(esm1b_batch_tokens, repr_layers=[12], return_contacts=True)

    # esm-msa-1b sequence representation
    token_representations = results["representations"][12].mean(1)


    esm_msa_1d = token_representations[0, 1:, :]
    row_attentions = results['row_attentions'].squeeze()[:, :, 1:, 1:]
    row_attentions = row_attentions.reshape(144, row_attentions.shape[-2], row_attentions.shape[-1])

    # print(token_representations.shape, esm_msa_1d.shape, row_attentions.shape)

    if "cuda" in args.device:
        esm_msa_1d = esm_msa_1d.cpu().numpy()
        row_attentions = row_attentions.cpu().numpy()

    data = {'esm_msa_1d': esm_msa_1d, 'row_attentions': row_attentions}

    # save into pkl file
    with open(args.feat_path, 'wb') as fw:
        pickle.dump(data, fw)


def load_esm(path):

    model, alphabet = esm.pretrained.load_model_and_alphabet(path)
    model = model.eval()
    batch_converter = alphabet.get_batch_converter()

    return model, batch_converter

def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    # translation for read sequence
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)

    return sequence.translate(translation)


if __name__ == "__main__":
    main()


