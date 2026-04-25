"""
Preprocessing script for raw experiment data.

Reads:  data/experiment_data_with_stimuli_for_R.csv  (contains Prolific IDs — never commit)
Writes:
  data/experiment_data_anonymised.csv   trial-level data with Prolific IDs replaced by P001, P002, ...
  data/iRT_AJT.csv                      item-level summary (one row per item) with iRT_AJT and stimuli vars
"""

from pathlib import Path
import pandas as pd

RAW = Path('data/experiment_data_with_stimuli_for_R.csv')
OUT_TRIAL = Path('data/experiment_data_anonymised.csv')
OUT_ITEM  = Path('data/iRT_AJT.csv')

if not RAW.exists():
    raise FileNotFoundError(f'Raw data not found: {RAW}')

df = pd.read_csv(RAW, index_col=0)
print(f'Loaded {len(df):,} rows, {df["ID"].nunique()} participants, {df["Item"].nunique()} items')

# --- Anonymise participant IDs ---
id_map = {pid: f'P{i+1:03d}' for i, pid in enumerate(df['ID'].unique())}
df['participant_id'] = df['ID'].map(id_map)
df = df.drop(columns=['ID'])

# Move participant_id to front
cols = ['participant_id'] + [c for c in df.columns if c != 'participant_id']
df = df[cols]

df.to_csv(OUT_TRIAL, index=False)
print(f'Saved anonymised trial data -> {OUT_TRIAL}  ({len(df):,} rows, {df["participant_id"].nunique()} participants)')

# --- Item-level summary ---
# iRT_AJT is already precomputed per item; take first row per item
ITEM_COLS = [
    'Item', 'Verb', 'Noun', 'Condition',
    'iRT_AJT', 'iAccuracy_AJT',
    'verb', 'noun', 'type',
    'noun_concreteness', 'snd3', 'snd10', 'snd25', 'snd50',
    'avg_v_sim_to_item', 'avg_v_sim_to_item_10nouns', 'avg_v_sim_to_item_50nouns',
    'verb_synset_len', 'logDice', 'collocation_freq',
    'VerbFreq', 'NounFreq',
]
existing = [c for c in ITEM_COLS if c in df.columns]
item_df = df.groupby('Item')[existing].first().reset_index(drop=True)
item_df = item_df.sort_values(['Condition', 'Item']).reset_index(drop=True)

item_df.to_csv(OUT_ITEM, index=False)
print(f'Saved item-level data     -> {OUT_ITEM}  ({len(item_df):,} items)')
print(f'\nCondition counts:\n{item_df["Condition"].value_counts().to_string()}')
print(f'\niRT_AJT range: {item_df["iRT_AJT"].min():.0f} – {item_df["iRT_AJT"].max():.0f} ms')
