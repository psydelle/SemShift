"""
Embed KWIC sentences using flair's TransformerWordEmbeddings with
sentence-transformers/all-MiniLM-L6-v2. Extracts contextual embeddings
for the verb and noun tokens (as identified by kwic_words) for each KWIC,
and saves everything to a .pt checkpoint.

Output .pt structure:
{
  "verb noun": {
    "verb": str,
    "noun": str,
    "kwics": List[str],
    "kwic_words": List[List[str, str]],   # [[verb_form, noun_form], ...]
    "verb_embeddings": Tensor[n_kwics, embed_dim],
    "noun_embeddings": Tensor[n_kwics, embed_dim],
    "failed_indices": List[int],          # KWICs where token lookup failed
  },
  ...
}
"""

import json
import click
import torch
from pathlib import Path
from tqdm import tqdm

from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def find_token_embedding(sentence: Sentence, word_form: str) -> torch.Tensor | None:
    """Return the embedding of the first token matching word_form (case-insensitive).
    If the word is split into sub-word pieces by the tokenizer, flair averages them
    into a single token embedding automatically.
    Returns None if not found.
    """
    word_lower = word_form.lower()
    for token in sentence:
        if token.text.lower() == word_lower:
            return token.embedding.clone()
    return None


@click.command()
@click.option("-j", "--input-json", required=True, help="Input JSON from build_comprehensive_kwic_dataset.py")
@click.option("-o", "--output-pt", required=True, help="Output .pt file path")
@click.option("--batch-size", default=64, show_default=True, help="Sentences per embedding batch")
@click.option("--checkpoint-every", default=50, show_default=True, help="Save partial checkpoint every N pairs")
@click.option("--device", default=None, help="Device: cpu / cuda / mps (default: auto-detect)")
def embed_kwics(input_json, output_pt, batch_size, checkpoint_every, device):
    """Embed verb and noun tokens in KWIC sentences using flair + all-MiniLM-L6-v2."""

    # ------------------------------------------------------------------ setup
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    import flair as flair_module
    flair_module.device = torch.device(device)

    print(f"Loading model: {MODEL_NAME}")
    embeddings = TransformerWordEmbeddings(
        MODEL_NAME,
        layers="-1",
        subtoken_pooling="mean",
        fine_tune=False,
    )

    # ----------------------------------------------------------- load inputs
    print(f"Loading KWICs from {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        kwics_json = json.load(f)
    print(f"  {len(kwics_json)} verb-noun pairs loaded")

    output_path = Path(output_pt)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from partial checkpoint if it exists
    if output_path.exists():
        results = torch.load(output_path, weights_only=False)
        print(f"  Resuming: {len(results)} pairs already embedded")
    else:
        results = {}

    # -------------------------------------------------------------- main loop
    pairs = [(k, v) for k, v in kwics_json.items() if k not in results]
    print(f"  {len(pairs)} pairs to embed\n")

    total_failed = 0

    for pair_idx, (pair_key, entry) in enumerate(tqdm(pairs, desc="Pairs")):
        verb = entry["verb"]
        noun = entry["noun"]
        kwics = entry["kwics"]
        kwic_words = entry["kwic_words"]  # [[verb_form, noun_form], ...]

        n = len(kwics)
        assert len(kwic_words) == n, f"kwics/kwic_words length mismatch for {pair_key}"

        # Build flair Sentence objects
        sentences = [Sentence(text) for text in kwics]

        # Embed in batches
        for i in range(0, n, batch_size):
            batch = sentences[i : i + batch_size]
            embeddings.embed(batch)

        # Extract verb and noun embeddings token-by-token
        verb_embs = []
        noun_embs = []
        failed_indices = []

        for i, (sentence, kw) in enumerate(zip(sentences, kwic_words)):
            verb_form, noun_form = kw[0], kw[1]

            v_emb = find_token_embedding(sentence, verb_form)
            n_emb = find_token_embedding(sentence, noun_form)

            if v_emb is None or n_emb is None:
                raise ValueError(f"Token embedding not found for pair '{pair_key}' in KWIC index {i}: verb='{verb_form}', noun='{noun_form}'")

            verb_embs.append(v_emb)
            noun_embs.append(n_emb)

        results[pair_key] = {
            "verb": verb,
            "noun": noun,
            "kwics": kwics,
            "kwic_words": kwic_words,
            "verb_embeddings": torch.stack(verb_embs).cpu(),
            "noun_embeddings": torch.stack(noun_embs).cpu(),
            "failed_indices": failed_indices,
        }

        if failed_indices:
            print(f"  [{pair_key}] {len(failed_indices)}/{n} KWICs had missing tokens")

        # Periodic checkpoint
        if (pair_idx + 1) % checkpoint_every == 0:
            torch.save(results, output_path)
            print(f"  Checkpoint saved ({len(results)} pairs)")

    # Final save
    torch.save(results, output_path)
    print(f"\nDone. {len(results)} pairs saved to {output_pt}")
    print(f"Total KWICs with failed token lookup: {total_failed}")


if __name__ == "__main__":
    embed_kwics()
