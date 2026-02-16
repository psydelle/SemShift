"""
Build a comprehensive KWIC dataset with:
1. Top 200 nouns per verb from SketchEngine
2. 100 KWICs for each verb-noun pair
3. Concreteness ratings and SND metrics for nouns
4. Count of nouns per verb for data completeness assessment
"""

import os
import json
import pandas as pd
import click
from tqdm import tqdm
from pathlib import Path
from nltk.corpus import wordnet as wn
from utils import get_vn_kwics, get_word_sketch, get_concreteness_rating, get_len_synset, load_snd_data

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

CORPUS = "preloaded/ententen21_tt31"


def load_existing_kwics(kwics_json_path):
    """Load existing KWICs from JSON file"""
    try:
        with open(kwics_json_path, 'r', encoding='utf-8') as f:
            existing_kwics = json.load(f)
        print(f"Loaded {len(existing_kwics)} existing KWIC entries from {kwics_json_path}")
        return existing_kwics
    except FileNotFoundError:
        print(f"No existing KWICs file found at {kwics_json_path}")
        return {}


def get_word_sketch_data(verb, noun):
    """Get word sketch data for a verb-noun pair"""
    try:
        ws_data = get_word_sketch(verb, noun, corpus=CORPUS)
        return {
            'coll_freq': ws_data.get('CollFreq', None),
            'logDice': ws_data.get('Score', None),  # Score is logDice
            'usage': ws_data.get('Usage', None),
        }
    except:
        return {'coll_freq': None, 'logDice': None, 'usage': None}


@click.command()
@click.option("-i", "--input-csv", required=True, help="Input CSV with stimuli (will extract unique verbs)")
@click.option("-o", "--output-csv", required=True, help="Output CSV path for comprehensive dataset")
@click.option("-j", "--output-json", required=True, help="Output JSON path for raw KWICs")
@click.option("-n", "--n-kwics", default=100, help="Number of KWICs per verb-noun pair")
@click.option("-c", "--n-ctx-sentences", default=1, help="Number of context sentences")
@click.option("-m", "--max-word-count", default=100, help="Max words per KWIC")
@click.option("--verbs", default=None, help="Comma-separated list of verbs to process (e.g., 'kick,kill,spill'). If None, processes all.")
@click.option("--nouns", default=None, help="Comma-separated list of nouns to process (e.g., 'ball,bunny,sponge'). If None, processes all.")
def build_dataset(input_csv, output_csv, output_json, n_kwics, n_ctx_sentences, max_word_count, verbs, nouns):
    """Build comprehensive KWIC dataset"""
    
    print("Loading data...")
    
    # Setup checkpoint
    checkpoint_dir = Path("data/processed/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / "comprehensive_kwic_checkpoint.json"
    
    # Load checkpoint if it exists
    if checkpoint_file.exists():
        print(f"Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        comprehensive_data = checkpoint_data.get('data', {})
        kwics_json = checkpoint_data.get('kwics_json', {})
        processed_pairs = set(checkpoint_data.get('processed_pairs', []))
        print(f"Resuming from {len(processed_pairs)} processed verb-noun pairs")
    else:
        comprehensive_data = {}
        kwics_json = {}
        processed_pairs = set()
    
    # Load unique verbs from input CSV
    stimuli_df = pd.read_csv(input_csv)
    all_verbs = stimuli_df["verb"].unique()
    all_nouns = stimuli_df["noun"].unique()
    
    # Define problematic verbs to exclude
    problematic_verbs = {'drain', 'muzzle', 'raise', 'spout', 'tackle'}
    filtered_verbs = [v for v in all_verbs if v not in problematic_verbs]
    
    # If specific verbs requested, filter to only those
    if verbs is not None:
        test_verbs = set(v.strip() for v in verbs.split(','))
        unique_verbs = [v for v in filtered_verbs if v in test_verbs]
        print(f"Testing with verbs: {list(unique_verbs)}")
    else:
        unique_verbs = filtered_verbs
        print(f"Processing {len(unique_verbs)} verbs (excluded: {list(problematic_verbs)})")
    
    print(f"Will process: {len(unique_verbs)} unique verbs and {len(all_nouns)} unique nouns from input CSV")
    
    # Load metadata
    snd_dict = load_snd_data()
    
    # data_to_process = [(verb, None) for verb in unique_verbs] + [(None, noun) for noun in all_nouns]
    # data_to_process = [(None, noun) for noun in all_nouns]
    data_to_process = [(None, "ball"), ("kick", None)]

    
    # Process each item (verb or noun) and get KWICs for all verb-noun pairs
    for verb, noun in tqdm(data_to_process, desc="Processing items"):
        verb_noun_count = 0
        
        # Get all paired items (nouns for verb, verbs for noun) (max 200 from SketchEngine)
        try:
            kwics_df = get_vn_kwics(
                CORPUS,
                verb=verb,
                noun=noun,
                n_kwics=n_kwics,
                n_ctx_sentences=n_ctx_sentences,
                max_word_count=max_word_count,
                subset_to_top_n_pairs=3
                # subset_to_top_n_pairs=9999  # Get all available pairs, we'll filter later
            )
            
            if kwics_df is not None and not kwics_df.empty:
                # Group by noun and process each verb-noun pair
                for (_verb, _noun), group in kwics_df.groupby(["verb", "noun"]):
                    if f"{_verb} {_noun}" in comprehensive_data:
                        if verb is not None:
                            comprehensive_data[f"{_verb} {_noun}"]['queried_by_verb'] = 1
                        elif noun is not None:
                            comprehensive_data[f"{_verb} {_noun}"]['queried_by_noun'] = 1
                        else:
                            raise ValueError("Both verb and noun cannot be None")
                    else:
                        # FILTER 1: Skip nouns less than 3 characters
                        if len(_noun) < 3:
                            continue
                        
                        kwics_list = group["kwics"].tolist()
                        kwic_words_list = group["kwic_words"].tolist()
                        
                        # Skip if we don't have enough KWICs
                        if len(kwics_list) != n_kwics:
                            print(f"Skipping {_verb}-{_noun}: only got {len(kwics_list)} KWICs")
                            continue
                        
                        # Get word sketch data for this verb-noun pair
                        ws_data = get_word_sketch_data(_verb, _noun)
                        
                        # Get concreteness for noun using existing utility function
                        noun_concreteness = get_concreteness_rating(_noun)
                        
                        # FILTER 2: Reject words if concreteness is not available (likely slang/informal)
                        if noun_concreteness is None or pd.isna(noun_concreteness):
                            continue
                        
                        # Get SND metrics for noun
                        snd_data = snd_dict.get(_noun.lower(), {})
                        
                        # Get synset lengths from WordNet
                        noun_synset_len = get_len_synset(_noun, pos=wn.NOUN)
                        verb_synset_len = get_len_synset(_verb, pos=wn.VERB)
                        
                        # Create comprehensive record
                        record = {
                            'verb': _verb,
                            'noun': _noun,
                            'item': f"{_verb} {_noun}",
                            'n_kwics': len(kwics_list),
                            'coll_freq': ws_data['coll_freq'],
                            'logDice': ws_data['logDice'],
                            'usage': ws_data['usage'],
                            'noun_concreteness': noun_concreteness,
                            'noun_synset_len': noun_synset_len,
                            'verb_synset_len': verb_synset_len,
                            'snd3': snd_data.get('snd3', None),
                            'snd10': snd_data.get('snd10', None),
                            'snd25': snd_data.get('snd25', None),
                            'snd50': snd_data.get('snd50', None),
                            "queried_by_verb": 1 if verb is not None else 0,
                            "queried_by_noun": 1 if noun is not None else 0,
                        }

                        kwics_record = {
                            'verb': _verb,
                            'noun': _noun,
                            'kwics': kwics_list,
                            'kwic_words': kwic_words_list,
                        }
                    
                        comprehensive_data[f"{_verb} {_noun}"] = record
                        verb_noun_count += 1
                        
                        # Store KWICs in JSON
                        kwics_json[f"{_verb} {_noun}"] = kwics_record
            
            print(f"{_verb}: {verb_noun_count} noun pairs with {n_kwics} KWICs each")
        
        except Exception as e:
            print(f"Error processing {_verb}: {str(e)}")
            continue
    
    # Convert to DataFrame and add noun count per verb
    if not comprehensive_data:
        print("No data collected!")
        return
    
    result_df = pd.DataFrame(list(comprehensive_data.values()))
    
    # Add noun count column
    noun_counts = result_df.groupby('verb').size().reset_index(name='n_nouns_for_verb')
    result_df = result_df.merge(noun_counts, on='verb')

    # Add verb count column
    verb_counts = result_df.groupby('noun').size().reset_index(name='n_verbs_for_noun')
    result_df = result_df.merge(verb_counts, on='noun')
    
    # Sort by verb and noun
    result_df = result_df.sort_values(['verb', 'noun']).reset_index(drop=True)
    
    # Save outputs. save all queried by verb to one CSV, and all queried by noun to another CSV
    print(f"\nSaving {len(result_df)} verb-noun pairs to CSV...")
    result_df[result_df['queried_by_verb'] == 1].to_csv(output_csv.replace(".csv", "_queried_by_verb.csv"), index=False)
    result_df[result_df['queried_by_noun'] == 1].to_csv(output_csv.replace(".csv", "_queried_by_noun.csv"), index=False)
    result_df.to_csv(output_csv, index=False)
    print(f"Saved to: {output_csv}")
    
    print(f"\nSaving raw KWICs to JSON...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(kwics_json, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {output_json}")
    
    # Clean up checkpoint on successful completion
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"\nCheckpoint file deleted (extraction completed successfully)")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total unique verbs: {result_df['verb'].nunique()}")
    print(f"Total verb-noun pairs: {len(result_df)}")
    print(f"Total KWICs: {len(result_df) * n_kwics}")
    print(f"\nNouns per verb:")
    print(result_df.groupby('verb')['n_nouns_for_verb'].first().describe())
    print(f"\nMissing values:")
    print(result_df.isnull().sum())


if __name__ == "__main__":
    build_dataset()
