"""
Build a comprehensive KWIC dataset with:
1. Top 200 nouns per verb or verbs per noun from SketchEngine
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
@click.option("--checkpoint-freq", default=1, help="Save checkpoint every N verbs/nouns processed (default: 1)")
@click.option("--checkpoint-dir", default="data/processed/checkpoints", help="Directory to save checkpoints")
def build_dataset(input_csv, output_csv, output_json, n_kwics, n_ctx_sentences, max_word_count, verbs, nouns, checkpoint_freq, checkpoint_dir):
    """Build comprehensive KWIC dataset"""
    
    print("\n" + "="*80)
    print("BUILDING COMPREHENSIVE KWIC DATASET")
    print("="*80)

    # ensure dirs for outputs exist
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    
    # Setup checkpoint
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / "comprehensive_kwic_checkpoint.json"
    
    # Load checkpoint if it exists
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        comprehensive_data = checkpoint_data.get('data', {})
        kwics_json = checkpoint_data.get('kwics_json', {})
        processed_items = set(tuple(item) for item in checkpoint_data.get('processed_items', []))
        print(f"\nResuming from checkpoint: {len(comprehensive_data)} pairs, {len(processed_items)} verbs/nouns processed")
    else:
        comprehensive_data = {}
        kwics_json = {}
        processed_items = set()
        print("\nStarting fresh (no checkpoint found)")
    
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
    else:
        unique_verbs = filtered_verbs
    
    # If specific nouns requested, filter to only those
    if nouns is not None:
        test_nouns = set(n.strip() for n in nouns.split(','))
        unique_nouns = [n for n in all_nouns if n in test_nouns]
    else:
        unique_nouns = list(all_nouns)
    
    # Build data_to_process based on what's specified
    data_to_process = []
    
    # Determine what to process based on user input
    if verbs is not None and nouns is None:
        data_to_process = [(verb, None) for verb in unique_verbs]
        query_mode = "verbs"
    elif nouns is not None and verbs is None:
        data_to_process = [(None, noun) for noun in unique_nouns]
        query_mode = "nouns"
    elif verbs is not None and nouns is not None:
        data_to_process = [(verb, None) for verb in unique_verbs] + [(None, noun) for noun in unique_nouns]
        query_mode = "verbs + nouns"
    else:
        data_to_process = [(verb, None) for verb in unique_verbs] + [(None, noun) for noun in unique_nouns]
        query_mode = "full dataset"
    
    # Print processing plan
    print(f"\nProcessing plan:")
    print(f"  Query mode: {query_mode}")
    print(f"  Verbs: {len(unique_verbs)} {'(excluded: ' + ', '.join(problematic_verbs) + ')' if verbs is None else ''}")
    print(f"  Nouns: {len(unique_nouns)}")
    print(f"  Total items to process: {len(data_to_process)}")
    print(f"  KWICs per pair: {n_kwics}")
    print(f"  Checkpoint frequency: every {checkpoint_freq} verb(s)/noun(s)")
    
    # Load metadata
    snd_dict = load_snd_data()

    # Checkpoint tracking
    items_processed_since_checkpoint = 0
    total_filtered_out = 0
    
    def save_checkpoint():
        """Save current progress to checkpoint file"""
        checkpoint_data = {
            'data': comprehensive_data,
            'kwics_json': kwics_json,
            'processed_items': list(processed_items)
        }
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False)
        print(f"    ✓ Checkpoint saved: {len(comprehensive_data)} pairs, {len(processed_items)} verbs/nouns")
    
    # Process each item (verb or noun) and get KWICs for all verb-noun pairs
    print("\nStarting processing...\n")
    for verb, noun in tqdm(data_to_process, desc="Processing"):
        # Skip if already processed (from checkpoint)
        item_key = (verb, noun)
        if item_key in processed_items:
            continue
        
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
                # subset_to_top_n_pairs=200,
                # subset_to_top_n_pairs=9999  # Get all available pairs, we'll filter later
            )
            
            if kwics_df is not None and not kwics_df.empty:
                # Group by noun and process each verb-noun pair
                for (_verb, _noun), group in kwics_df.groupby(["verb", "noun"]):
                    pair_key = f"{_verb} {_noun}"
                    
                    # Skip if already processed (from checkpoint)
                    if pair_key in comprehensive_data:
                        # Just update query flags if needed
                        if verb is not None:
                            comprehensive_data[pair_key]['queried_by_verb'] = 1
                        elif noun is not None:
                            comprehensive_data[pair_key]['queried_by_noun'] = 1
                        continue
                    
                    # FILTER 1: Skip nouns less than 3 characters
                    if len(_noun) < 3:
                        total_filtered_out += 1
                        continue
                    
                    kwics_list = group["kwics"].tolist()
                    kwic_words_list = group["kwic_words"].tolist()
                    
                    # Skip if we don't have enough KWICs
                    if len(kwics_list) != n_kwics:
                        total_filtered_out += 1
                        continue
                    
                    # Get word sketch data for this verb-noun pair
                    ws_data = get_word_sketch_data(_verb, _noun)
                    
                    # Get concreteness for noun using existing utility function
                    noun_concreteness = get_concreteness_rating(_noun)
                    
                    # FILTER 2: Reject words if concreteness is not available (likely slang/informal)
                    if noun_concreteness is None or pd.isna(noun_concreteness):
                        total_filtered_out += 1
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
                
                    comprehensive_data[pair_key] = record
                    verb_noun_count += 1
                    
                    # Store KWICs in JSON
                    kwics_json[pair_key] = kwics_record
            
            # Mark this verb/noun as processed
            processed_items.add(item_key)
            items_processed_since_checkpoint += 1
            
            # Save checkpoint periodically (after processing each verb/noun)
            if items_processed_since_checkpoint >= checkpoint_freq:
                save_checkpoint()
                items_processed_since_checkpoint = 0
        
        except Exception as e:
            print(f"  ✗ Error processing {verb} {noun}: {str(e)}")
            raise
    
    # Final checkpoint save before finishing
    print(f"\n{'='*80}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"  Items processed: {len(processed_items)}")
    print(f"  Pairs collected: {len(comprehensive_data)}")
    print(f"  Pairs filtered: {total_filtered_out}")
    
    if items_processed_since_checkpoint > 0:
        save_checkpoint()
    
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
    
    # Save outputs
    print(f"\n{'='*80}")
    print(f"SAVING OUTPUTS")
    print(f"{'='*80}")
    
    result_df[result_df['queried_by_verb'] == 1].to_csv(output_csv.replace(".csv", "_queried_by_verb.csv"), index=False)
    result_df[result_df['queried_by_noun'] == 1].to_csv(output_csv.replace(".csv", "_queried_by_noun.csv"), index=False)
    result_df.to_csv(output_csv, index=False)
    print(f"  ✓ CSV saved: {output_csv}")
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(kwics_json, f, ensure_ascii=False, indent=2)
    print(f"  ✓ JSON saved: {output_json}")
    
    # Clean up checkpoint on successful completion
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"  ✓ Checkpoint deleted")
    
# print summary of results filter queried by verb 
    print(f"\n{'='*80}")
    print(f"SUMMARY BY VERB")
    print(f"{'='*80}")
    queried_by_verb = result_df[result_df['queried_by_verb'] == 1]
    print(f"  Verbs queried: {queried_by_verb['verb'].nunique()}")
    print(f"  Mean pairs per verb: {queried_by_verb.groupby('verb').size().mean():.1f}")

        # print summary of results filter queried by noun
    print(f"\n{'='*80}")
    print(f"SUMMARY BY NOUN")
    print(f"{'='*80}")
    queried_by_noun = result_df[result_df['queried_by_noun'] == 1]
    print(f"  Nouns queried: {queried_by_noun['noun'].nunique()}")
    print(f"  Mean pairs per noun: {queried_by_noun.groupby('noun').size().mean():.1f}")



if __name__ == "__main__":
    build_dataset()


# To run this script, use the command line with appropriate arguments, for example:
# uv run python build_comprehensive_kwic_dataset.py -i stimuli.csv -o data/processed/comprehensive_kwic_dataset.csv -j data/processed/comprehensive_kwic_dataset.json --verbs "kick,kill" --nouns "ball,bunny" --checkpoint-freq 1