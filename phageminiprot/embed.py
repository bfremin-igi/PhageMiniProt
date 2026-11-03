"""Generate mean-pooled ESM-2 embeddings from a protein FASTA."""

import pathlib

import click
import esm
import numpy as np
import torch
import tqdm
from Bio import SeqIO

MODEL_OPTIONS = {
    "esm2_t48_15B_UR50D": 5120,
    "esm2_t36_3B_UR50D": 2560,
    "esm2_t33_650M_UR50D": 1280,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t6_8M_UR50D": 320,
}

DEFAULT_TOKS_PER_BATCH = 4096


def embed_fasta(
    fasta_file,
    output_filepath,
    model_name="esm2_t6_8M_UR50D",
    layer=-1,
    toks_per_batch=DEFAULT_TOKS_PER_BATCH,
):
    """Embed sequences and write a (n_sequences, dim) .npy array in FASTA order."""
    output_filepath = pathlib.Path(output_filepath)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval().to(device)

    dataset = esm.FastaBatchedDataset.from_file(fasta_file)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(1022),
        batch_sampler=dataset.get_batch_indices(
            toks_per_batch=toks_per_batch, extra_toks_per_seq=1
        ),
    )

    if layer == -1:
        layer = model.num_layers

    sequence_ids, mean_embeddings = [], []
    with torch.no_grad():
        for headers, sequences, tokens in tqdm.tqdm(dataloader, total=len(dataloader)):
            tokens = tokens.to(device)
            outputs = model(tokens, repr_layers=[layer], return_contacts=False)
            embeddings = outputs["representations"][layer]

            for i, (header, sequence) in enumerate(zip(headers, sequences)):
                valid_length = min(len(sequence), 1022)
                mean_embedding = (
                    embeddings[i, 1 : valid_length + 1].mean(dim=0).cpu().numpy()
                )
                sequence_ids.append(header.split()[0])
                mean_embeddings.append(mean_embedding)

    mean_embeddings = np.stack(mean_embeddings)
    fasta_ids = [record.id for record in SeqIO.parse(fasta_file, "fasta")]
    id_to_index = {seq_id: i for i, seq_id in enumerate(sequence_ids)}
    missing = [fid for fid in fasta_ids if fid not in id_to_index]
    if missing:
        raise ValueError(
            f"Embedding output is missing {len(missing)} FASTA id(s); "
            f"first missing: {missing[0]!r}"
        )
    reordered = mean_embeddings[[id_to_index[fid] for fid in fasta_ids]]
    np.save(output_filepath, reordered)
    return reordered


@click.command()
@click.option(
    "--output-filepath",
    type=click.Path(),
    required=True,
    help="Path to save embeddings (.npy).",
)
@click.argument("fasta_file", type=click.Path(exists=True))
@click.option(
    "--model",
    type=click.Choice(list(MODEL_OPTIONS.keys())),
    default="esm2_t6_8M_UR50D",
    help="ESM-2 model variant.",
)
@click.option("--layer", type=int, default=-1, help="Layer index (-1 = final).")
@click.option(
    "--toks-per-batch",
    type=int,
    default=DEFAULT_TOKS_PER_BATCH,
    show_default=True,
    help="Approximate tokens per embedding batch.",
)
def generate_embeddings(output_filepath, fasta_file, model, layer, toks_per_batch):
    """Generate embeddings for sequences in a FASTA file."""
    embed_fasta(
        fasta_file=fasta_file,
        output_filepath=output_filepath,
        model_name=model,
        layer=layer,
        toks_per_batch=toks_per_batch,
    )


if __name__ == "__main__":
    generate_embeddings()
