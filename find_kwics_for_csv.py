import os
import time
import requests as r
import pandas as pd
import math
import click
from dotenv import load_dotenv
from tqdm import tqdm
import json
import re
from utils import get_vn_kwics

# Load environment variables from .env
load_dotenv()

# Predefined corpus path (could be modified)
CORPUS_EN = "preloaded/ententen21_tt31"

# Helper function to process each verb-noun line
def process_item(line, n_kwics, n_ctx_sentences, max_word_count, n_nouns_per_verb=None):
    # Use utility function to get KWICs for the verb and noun
    data = get_vn_kwics(
        CORPUS_EN,
        line["verb"],
        line["noun"],
        n_kwics=n_kwics,
        n_ctx_sentences=n_ctx_sentences,
        max_word_count=max_word_count,
        subset_to_top_n_pairs=n_nouns_per_verb
    )
    return data

# Function that handles timeout and prints when no KWICs are found
def process_line(line, **kwargs):
    p = process_item(line, **kwargs)
    if p.empty:
        print("No KWICS found for", line["verb"], line["noun"])
    time.sleep(10)  # Mitigate SketchEngine API rate limits
    return p

# Helper function to load an existing JSON file or start fresh if corrupted/missing
def load_json_file(out_file):
    try:
        with open(out_file, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

@click.command()
@click.option("-f", "--csv-file", required=True, help="CSV file with verbs and nouns")
@click.option("-o", "--out-file", required=True, help="Output JSON file path")
@click.option("-n", "--n-kwics", default=100, help="Number of KWICs to fetch for each verb-noun pair")
@click.option("-c", "--n-ctx-sentences", default=1, help="Number of sentences before and after the sentence with the KWIC")
@click.option("-m", "--max-word-count", default=100, help="Maximum number of words in the KWIC")
@click.option("--n-nouns-per-verb", type=int, default=None, help="Number of top nouns by score for each verb")
def process_corpus(csv_file, out_file, n_kwics, n_ctx_sentences, max_word_count, n_nouns_per_verb):
    # Load the input CSV file into a DataFrame
    in_df = pd.read_csv(csv_file)

    # If n_nouns_per_verb is provided, ensure only unique verbs are processed
    if n_nouns_per_verb is not None:
        in_df = in_df.drop_duplicates(subset=["verb"])

    # Load already processed data from JSON file (handle corrupted/missing files)
    processed_data = load_json_file(out_file)
    if processed_data:
        print(f"Loaded {len(processed_data)} processed items from {out_file}")
    else:
        processed_data = {}

    # Iterate over each line in the input DataFrame
    for idx, line in tqdm(in_df.iterrows(), total=len(in_df)):
        verb = line["verb"]
        skip = False

        # Skip already processed verbs (or verb-noun pairs if n_nouns_per_verb is None)
        for key, value in processed_data.items():
            if verb == value["verb"]:
                if n_nouns_per_verb is not None:
                    print(f"Skipping {verb} as it is already processed")
                    skip = True
                    break
                else:
                    # Skip if both verb and noun are already processed
                    noun = line["noun"]
                    if noun == value["noun"]:
                        print(f"Skipping {verb} {noun} as it is already processed")
                        skip = True
                        break

        if skip:
            continue  # Skip the rest of the iteration if already processed

        # Process the line to get KWIC data
        line_output_df = process_line(
            line, n_kwics=n_kwics, n_ctx_sentences=n_ctx_sentences, max_word_count=max_word_count, n_nouns_per_verb=n_nouns_per_verb
        )

        # Create an "item" by concatenating verb and noun
        line_output_df["item"] = line_output_df["verb"] + " " + line_output_df["noun"]

        # Group by "item", aggregate the data for each verb-noun pair
        line_output_df = line_output_df.groupby("item").agg(
            {
                "kwics": lambda x: list(x),
                "kwic_words": lambda x: list(x),
                "verb": "first",
                "noun": "first",
            }
        )

        # Reset the index
        line_output_df = line_output_df.reset_index()

        # Set "item" as the index
        line_output_df = line_output_df.set_index("item")

        # Convert DataFrame to a dictionary (ready for JSON)
        data = line_output_df.to_dict(orient="index")

        # Update processed data with the new entries
        processed_data.update(data)

        # Write to a temporary file first to avoid corruption in case of interruption
        temp_file = out_file + ".tmp"
        with open(temp_file, "w", errors='ignore') as f:
            json.dump(processed_data, f, ensure_ascii=False)

        # Safely replace the original file with the new data after successful write
        os.replace(temp_file, out_file)

    print("Processing complete!")

if __name__ == "__main__":
    process_corpus()
