import json
import os
from pathlib import Path
import pickle
from typing import List, Optional
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import torch
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings, WordEmbeddings
import seaborn as sns
from scipy.stats import wasserstein_distance

def get_embedding(
    model, contexts: List[str], context_words: List[List[str]], do_concat_tokens=False
):
    
    '''Get the embeddings for the kwics using the specified model.'''
    embeddings = []

    for cxt, cxt_words in zip(contexts, context_words):
        # Create a Sentence object
        sentence = Sentence(cxt)
        # Embed the sentence using the provided model
        model.embed(sentence)

        # Collect the embeddings for the context words in order
        _embeddings = []
        # match verb first, and closest instance of noun after verb
        verb, noun = cxt_words
        verb_found = False

        # for token in sentence:
        #     if token.text == verb.lower():
        #         verb_found = True
        #         _embeddings.append(token.embedding)
        #     elif verb_found and token.text == noun.lower():
        #         _embeddings.append(token.embedding)
        #         break
        #
        def _match_words(sentence, verb, noun):
            """Match the embeddings of the verb and noun in the sentence
            Args:
                sentence: a list of tokens
                verb: the verb to match
                noun: the noun to match
            Returns:
                word1_id: the index of the verb in the sentence
                word2_id: the index of the noun in the sentence

            Potential issues: The code may not work as expected if the verb is repeated in adjetival form,
            e.g., "They abandoned the abandoned dog."
            """
            word1_id, word2_id = None, None
            for word1_id in range(len(sentence)):
                word1 = sentence[word1_id].text
                if word1 == verb:
                    for offset in range(1, 10):
                        word2_id = word1_id + offset
                        # attempt to match second word
                        if word2_id < len(sentence) and sentence[word2_id].text == verb:
                            break  # skip if verb is repeated, because it is not the proximal verb
                        elif word2_id < len(sentence) and sentence[word2_id].text == noun:
                            return word1_id, word2_id
            # If no match found, return None
            return None

        result = _match_words(sentence, verb, noun)
        
        if result is None:
            print(f"Warning: Could not match '{verb}' and '{noun}' in context")
            print(f"Sentence tokens: {[token.text for token in sentence]}")
            # Return zero embeddings or skip this kwic
            _embeddings = [torch.zeros(sentence[0].embedding.shape), torch.zeros(sentence[0].embedding.shape)]
        else:
            word1_id, word2_id = result
            _embeddings = [sentence[word1_id].embedding, sentence[word2_id].embedding]

        # Ensure that we have embeddings for exactly the number of context words expected
        assert len(_embeddings) == len(
            cxt_words
        ), f"Mismatch in lengths: {_embeddings} vs {cxt_words}"

        # Stack the embeddings of the current context words
        embeddings.append(torch.stack(_embeddings))

    # Stack the embeddings across all contexts and take the mean across them
    word_embeddings = torch.stack(embeddings, dim=0).mean(dim=0)

    if do_concat_tokens:
        # If concatenation is requested, flatten the tensor
        word_tokens_output = word_embeddings.reshape(-1)
    else:
        # Otherwise, take the mean across the embeddings
        word_tokens_output = torch.mean(word_embeddings, dim=0)

    return word_tokens_output


def get_fasttext_vector(contexts, context_words, model, do_concat_tokens=False):
    return get_embedding(model, contexts, context_words, do_concat_tokens)


def get_transformer_vector(contexts, context_words, model, do_concat_tokens=True):
    return get_embedding(model, contexts, context_words, do_concat_tokens)


def get_embeddings(
    dataset: pd.DataFrame,
    kwics: dict,
    model_name: str,
    do_concat_tokens: bool,
    avg_last_n_layers: int,
    checkpoint_name: Optional[str] = None
):
    '''Get embeddings for the dataset using the specified model and parameters.
    Supports incremental checkpointing to resume from crashes.
    
    Args:
        dataset: a DataFrame containing the dataset
        kwics: a dictionary containing the KWICs for each item
        model_name: the name of the model to use
        do_concat_tokens: a boolean indicating whether to concatenate the tokens
        avg_last_n_layers: the number of layers to average over
        checkpoint_name: optional name for checkpoint file (e.g., "10nouns", "50nouns")
    Returns:
        colloc2BERT: a dictionary containing the embeddings for each item
    '''
    
    # Setup checkpoint directory and file
    checkpoint_dir = Path("data/processed/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if checkpoint_name:
        checkpoint_file = checkpoint_dir / f"embeddings_checkpoint_{checkpoint_name}.json"
    else:
        checkpoint_file = checkpoint_dir / "embeddings_checkpoint.json"
    
    # Load existing checkpoint if available
    if checkpoint_file.exists():
        print(f"Loading existing checkpoint from {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        completed_items = set(checkpoint_data.get('completed_items', []))
        colloc2BERT = checkpoint_data.get('embeddings', {})
        # Convert list vectors back to tensors
        for item in colloc2BERT:
            if isinstance(colloc2BERT[item]['vec'], list):
                colloc2BERT[item]['vec'] = torch.tensor(colloc2BERT[item]['vec'])
        print(f"Resuming from {len(completed_items)} completed items")
    else:
        completed_items = set()
        colloc2BERT = dict()
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu"

    if model_name == "sbert":
        BERT = "sentence-transformers/all-MiniLM-L6-v2"
        layers=",".join([str(i) for i in range(1, avg_last_n_layers + 1)])
        model = TransformerWordEmbeddings(
            BERT, subtoken_pooling="mean", layer_mean=True, layers=layers
        ).to(device)


        for item in dataset["item"]:
            # Skip if already completed
            if item in completed_items:
                print(f"Skipping {item} (already completed)")
                continue
                
            if kwics and kwics.get(item, {}).get("kwics"):
                colloc_kwics = kwics[item]["kwics"]
                colloc_kwics_words = kwics[item]["kwic_words"]
                n_kwics = len(colloc_kwics)
                type = dataset[dataset["item"] == item]["type"].values[0]
            else:
                print(f"Warning: No kwics provided for {item}, skipping")
                completed_items.add(item)
                continue

            print(f"\nItem: {item}")
            print(f"Type: {type}")
            print(f"Number of KWICs: {n_kwics}")

            print(f'Retrieving vector for "{item}" from {BERT}')
            vec = get_transformer_vector(
                colloc_kwics,
                colloc_kwics_words,
                model,
                do_concat_tokens=do_concat_tokens,
            )

            print(f"Vector shape for {item}: {vec.shape}")

            # Store embedding
            colloc2BERT[item] = {"vec": vec.to("cpu"), "n_kwics": n_kwics, "type": type}
            completed_items.add(item)
            
            # Save checkpoint after each item
            checkpoint_data = {
                'completed_items': list(completed_items),
                'embeddings': {k: {**v, 'vec': v['vec'].tolist()} if isinstance(v['vec'], torch.Tensor) else v for k, v in colloc2BERT.items()}
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            print(f"Checkpoint saved ({len(completed_items)} items completed)")

    elif model_name == "fasttext":
        MODEL = "en-crawl"
        model = WordEmbeddings(MODEL).to(device) # todo: if no .to(), just remove it

        for item in dataset["item"]:
            # Skip if already completed
            if item in completed_items:
                print(f"Skipping {item} (already completed)")
                continue
                
            colloc_kwics = kwics[item]["kwics"]
            colloc_kwics_words = kwics[item]["kwic_words"]
            n_kwics = len(colloc_kwics)
            type = dataset[dataset["item"] == item]["type"].values[0]

            print(f"\nItem: {item}")
            print(f"Type: {type}")
            print(f"Number of KWICs: {n_kwics}")

            print(f'Retrieving vector for "{item}" from {model_name}')
            vec = get_fasttext_vector(
                colloc_kwics,
                colloc_kwics_words,
                model,
                do_concat_tokens=do_concat_tokens,
            )

            print(f"Vector shape for {item}: {vec.shape}")

            colloc2BERT[item] = {"vec": vec.to("cpu"), "n_kwics": n_kwics, "type": type}
            completed_items.add(item)
            
            # Save checkpoint after each item
            checkpoint_data = {
                'completed_items': list(completed_items),
                'embeddings': {k: {**v, 'vec': v['vec'].tolist()} if isinstance(v['vec'], torch.Tensor) else v for k, v in colloc2BERT.items()}
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            print(f"Checkpoint saved ({len(completed_items)} items completed)")

    # Build avg verb embeddings
    df = pd.DataFrame.from_dict(colloc2BERT, orient="index")
    df = df.reset_index() 
    df = df.rename(columns={"index": "item"})
    # Drop n_kwics from df since we already have it in dataset
    df = df.drop(columns=['n_kwics'], errors='ignore')
    dataset = dataset.merge(df, on="item")

    # Group by verb and average the embeddings
    verb2BERT = dict()
    for verb, group in dataset.groupby("verb"):
        # Convert to tensors if they're lists (from checkpoint loading)
        vecs = [torch.tensor(v) if isinstance(v, list) else v for v in group["vec"].values]
        vecs = torch.stack(vecs)
        avg_vec = torch.mean(vecs, dim=0)
        # Use n_kwics from dataset, or from group if it exists
        n_kwics_total = group["n_kwics"].sum() if "n_kwics" in group.columns else len(group)
        verb2BERT[verb] = {"vec": avg_vec, 'n_kwics': n_kwics_total}
        print(f"Vector shape for {verb}: {avg_vec.shape}")
    
    # Clean up checkpoint file on successful completion
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"Checkpoint file deleted (embeddings completed successfully)")
        
    return colloc2BERT, verb2BERT


def train_and_evaluate_classifier(X, y, feature_name, seeds, label="classifier"):
    """
    Train and evaluate an SVM classifier using GridSearchCV and multiple random seeds.
    
    Args:
        X: Feature matrix
        y: Labels
        feature_name: Name of the feature set (for logging)
        seeds: List of random seeds to use
        label: Label for saving files
    
    Returns:
        Dictionary with results including best_model, accuracies, and predictions
    """
    print(f"\n{'='*100}")
    print(f"Training classifier with features: {feature_name}")
    print(f"{'='*100}")
    print(f"Feature shape: {X.shape}")
    
    # Define the parameter grid for GridSearch
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf']
    }

    seed_accuracies = []
    all_y_pred = None
    best_model = None

    for seed in seeds:
        print(f"\nRunning with seed {seed}...")

        n_splits = 5
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        grid_search = GridSearchCV(svm.SVC(), param_grid, cv=kf, scoring='accuracy', n_jobs=-1)

        accuracy_scores = []

        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        average_accuracy = np.mean(accuracy_scores)
        print(f"Average Accuracy for seed {seed}: {average_accuracy:.4f}")
        seed_accuracies.append(average_accuracy)

    best_model = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}")

    overall_average_accuracy = np.mean(seed_accuracies)
    overall_std = np.std(seed_accuracies)
    print(f"Overall Average Accuracy across all seeds: {overall_average_accuracy:.4f} (±{overall_std:.4f})")

    # Train on entire dataset for final predictions
    best_model.fit(X, y)
    entire_y_pred = best_model.predict(X)
    entire_accuracy = accuracy_score(y, entire_y_pred)
    print(f"Accuracy on entire dataset: {entire_accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y, entire_y_pred, target_names=best_model.classes_))

    return {
        'model': best_model,
        'feature_name': feature_name,
        'seed_accuracies': seed_accuracies,
        'overall_accuracy': overall_average_accuracy,
        'overall_std': overall_std,
        'entire_accuracy': entire_accuracy,
        'predictions': entire_y_pred,
        'y_true': y
    }


def train_and_evaluate_classifier(X, y, feature_name, seeds, label="classifier"):
    """
    Train and evaluate an SVM classifier using GridSearchCV and multiple random seeds.
    
    Args:
        X: Feature matrix
        y: Labels
        feature_name: Name of the feature set (for logging)
        seeds: List of random seeds to use
        label: Label for saving files
    
    Returns:
        Dictionary with results including best_model, accuracies, and predictions
    """
    print(f"\n{'='*100}")
    print(f"Training classifier with features: {feature_name}")
    print(f"{'='*100}")
    print(f"Feature shape: {X.shape}")
    
    # Define the parameter grid for GridSearch
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf']
    }

    seed_accuracies = []
    best_model = None

    for seed in seeds:
        print(f"\nRunning with seed {seed}...")

        n_splits = 5
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        grid_search = GridSearchCV(svm.SVC(), param_grid, cv=kf, scoring='accuracy', n_jobs=-1)

        accuracy_scores = []

        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        average_accuracy = np.mean(accuracy_scores)
        print(f"Average Accuracy for seed {seed}: {average_accuracy:.4f}")
        seed_accuracies.append(average_accuracy)

    best_model = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}")

    overall_average_accuracy = np.mean(seed_accuracies)
    overall_std = np.std(seed_accuracies)
    print(f"Overall Average Accuracy across all seeds: {overall_average_accuracy:.4f} (±{overall_std:.4f})")

    # Train on entire dataset for final predictions
    best_model.fit(X, y)
    entire_y_pred = best_model.predict(X)
    entire_accuracy = accuracy_score(y, entire_y_pred)
    print(f"Accuracy on entire dataset: {entire_accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y, entire_y_pred, target_names=best_model.classes_))

    return {
        'model': best_model,
        'feature_name': feature_name,
        'seed_accuracies': seed_accuracies,
        'overall_accuracy': overall_average_accuracy,
        'overall_std': overall_std,
        'entire_accuracy': entire_accuracy,
        'predictions': entire_y_pred,
        'y_true': y
    }


def main():
    dataset = "data/stimuli/stimuli_no_fillers_freq_data.csv"
    dataset_path_with_sims = "data/stimuli/stimuli_no_fillers_freq_data_sims.csv"
    dataset_to_use = pd.read_csv(dataset)
    avg_last_n_layers = 1
    kwics_specific_nouns = json.load(open("data/stimuli/kwics_stimuli_no_fillers.json"))
    kwics_10_nouns = json.load(open("data/stimuli/kwics_stimuli_no_fillers_10_nouns.json"))
    kwics_50_nouns = json.load(open("data/stimuli/kwics_stimuli_no_fillers_50_nouns.json"))
    do_concat_tokens = True
    label = "avg_verb"  # optional
    model_name = "sbert"
    seeds = [42, 123, 2023, 7, 90]  # List of seeds


    colloc_embeddings_cache_filename = f'data/processed/{model_name}_{Path(dataset).name[:-4]}-last_{avg_last_n_layers}-kwics-{"-concat" if do_concat_tokens else ""}{"-" + label if label else ""}.dat'
    os.makedirs(os.path.dirname(colloc_embeddings_cache_filename), exist_ok=True)
    if not os.path.isfile(colloc_embeddings_cache_filename):
        colloc_item_embeddings, colloc_verb_embeddings = get_embeddings(
            dataset_to_use, kwics_specific_nouns, model_name, do_concat_tokens, avg_last_n_layers,
            checkpoint_name="main_dataset"
        )
        # write the embeddings dictionary to a file to be re-used next time we run the code
        with open(colloc_embeddings_cache_filename, "wb") as colloc2BERTfile:
            pickle.dump((colloc_item_embeddings, colloc_verb_embeddings), colloc2BERTfile)
        print("Dictionary written to file\n")

    else:
        # get the previously calculated embeddings from the file in which they were stored
        with open(colloc_embeddings_cache_filename, "rb") as colloc2BERTfile:
            colloc_item_embeddings, colloc_verb_embeddings = pickle.load(colloc2BERTfile)
        print(f"Read from file {colloc_embeddings_cache_filename}")

    # get verb embeddings for 10 nouns paired with the verb
    concat_str = "-concat" if do_concat_tokens else ""
    verb_embeddings_cache_filename_10 = f'data/processed/{model_name}_{Path(dataset).name[:-4]}-last_{avg_last_n_layers}-kwics{concat_str}-verb-10nouns.dat'
    if not os.path.isfile(verb_embeddings_cache_filename_10):
        # make fake dataset with all verbs and 10 nouns, as they appear in kwics_10_nouns
        _df = []
        for item, data in kwics_10_nouns.items():
            _df.append(
                {
                    "verb": data["verb"],
                    "item": item,
                    "type": "unknown",
                    "noun": data["noun"],
                }
            )
        _df = pd.DataFrame(_df)
        print(_df.groupby("verb").count().sort_values("item", ascending=False))
        verb_embeddings_by_noun_10, avg_all_noun_verb_embeddings_10 = get_embeddings(
            _df, kwics_10_nouns, model_name, do_concat_tokens, avg_last_n_layers,
            checkpoint_name="10nouns"
        )
        # write the embeddings dictionary to a file to be re-used next time we run the code
        with open(verb_embeddings_cache_filename_10, "wb") as verb2BERTfile:
            pickle.dump((verb_embeddings_by_noun_10, avg_all_noun_verb_embeddings_10), verb2BERTfile)
        print("Dictionary written to file\n")

    else:
        # get the previously calculated embeddings from the file in which they were stored
        with open(verb_embeddings_cache_filename_10, "rb") as verb2BERTfile:
            verb_embeddings_by_noun_10, avg_all_noun_verb_embeddings_10 = pickle.load(verb2BERTfile)
        print(f"Read from file {verb_embeddings_cache_filename_10}")

    # get verb embeddings for 50 nouns paired with the verb
    verb_embeddings_cache_filename_50 = f'data/processed/{model_name}_{Path(dataset).name[:-4]}-last_{avg_last_n_layers}-kwics{concat_str}-verb-50nouns.dat'
    if not os.path.isfile(verb_embeddings_cache_filename_50):
        # make fake dataset with all verbs and 50 nouns, as they appear in kwics_50_nouns
        _df = []
        for item, data in kwics_50_nouns.items():
            _df.append(
                {
                    "verb": data["verb"],
                    "item": item,
                    "type": "unknown",
                    "noun": data["noun"],
                }
            )
        _df = pd.DataFrame(_df)
        print(_df.groupby("verb").count().sort_values("item", ascending=False))
        verb_embeddings_by_noun_50, avg_all_noun_verb_embeddings_50 = get_embeddings(
            _df, kwics_50_nouns, model_name, do_concat_tokens, avg_last_n_layers,
            checkpoint_name="50nouns"
        )
        # write the embeddings dictionary to a file to be re-used next time we run the code
        with open(verb_embeddings_cache_filename_50, "wb") as verb2BERTfile:
            pickle.dump((verb_embeddings_by_noun_50, avg_all_noun_verb_embeddings_50), verb2BERTfile)
        print("Dictionary written to file\n")

    else:
        # get the previously calculated embeddings from the file in which they were stored
        with open(verb_embeddings_cache_filename_50, "rb") as verb2BERTfile:
            verb_embeddings_by_noun_50, avg_all_noun_verb_embeddings_50 = pickle.load(verb2BERTfile)
        print(f"Read from file {verb_embeddings_cache_filename_50}")


    embed_dim = 384 if model_name == "sbert" else 300
    if do_concat_tokens:
        embed_dim *= 2

    # create a dataframe from the embeddings dictionary
    noun_df = pd.DataFrame.from_dict(colloc_item_embeddings, orient="index")
    noun_df = noun_df.reset_index()
    noun_df = noun_df.rename(columns={"index": "item"})

    verb_df = pd.DataFrame.from_dict(colloc_verb_embeddings, orient="index")
    verb_df = verb_df.reset_index()
    verb_df = verb_df.rename(columns={"index": "verb"})

    all_nouns_10_verb_df = pd.DataFrame.from_dict(avg_all_noun_verb_embeddings_10, orient="index")
    all_nouns_10_verb_df = all_nouns_10_verb_df.reset_index()
    all_nouns_10_verb_df = all_nouns_10_verb_df.rename(columns={"index": "verb", "vec": "vec_avg_verb_10nouns"})

    all_nouns_50_verb_df = pd.DataFrame.from_dict(avg_all_noun_verb_embeddings_50, orient="index")
    all_nouns_50_verb_df = all_nouns_50_verb_df.reset_index()
    all_nouns_50_verb_df = all_nouns_50_verb_df.rename(columns={"index": "verb", "vec": "vec_avg_verb_50nouns"})


    # get cosine similarity between verb embedding and item embedding
    join = dataset_to_use.merge(noun_df[["item", "vec"]], on="item")
    join = join.merge(verb_df[["verb", "vec"]], on="verb", suffixes=("_item", "_avg_verb"))
    join = join.merge(all_nouns_10_verb_df[["verb", "vec_avg_verb_10nouns"]], on="verb") 
    join = join.merge(all_nouns_50_verb_df[["verb", "vec_avg_verb_50nouns"]], on="verb")

    # calculate cosine similarity using torch
    join["avg_v_sim_to_item"] = join.apply(
        lambda x: torch.nn.functional.cosine_similarity(x["vec_item"], x["vec_avg_verb"], dim=0).item(),
        axis=1,
    )
 
 # calculate cosine similarity using torch
    join["avg_v_sim_to_item_10nouns"] = join.apply(
        lambda x: torch.nn.functional.cosine_similarity(x["vec_item"], x["vec_avg_verb_10nouns"], dim=0).item(),
        axis=1,
    )

    # calculate cosine similarity using torch for 50 nouns
    join["avg_v_sim_to_item_50nouns"] = join.apply(
        lambda x: torch.nn.functional.cosine_similarity(x["vec_item"], x["vec_avg_verb_50nouns"], dim=0).item(),
        axis=1,
    )

    # Load SND metrics from reilly_desai file
    snd_df = pd.read_csv("data/stimuli/snd_reilly_desai.txt", sep=r'\s+', engine='python')
    snd_df.columns = ['word', 'PoS', 'snd3', 'snd10', 'snd25', 'snd50']
    
    # Filter for nouns only (PoS='NN')
    snd_df = snd_df[snd_df['PoS'] == 'NN']
    
    # Extract noun names from join and merge with SND data
    # Match nouns with the reilly_desai word column
    join = join.merge(snd_df[['word', 'snd3', 'snd10', 'snd25', 'snd50']], 
                      left_on='noun', right_on='word', how='left')
    join = join.drop(columns=['word'])
    
    print(f"SND columns added. Join shape: {join.shape}")
    print(f"Columns in join: {list(join.columns)}")

    # # Calculate the Wasserstein distance between the item and verb embeddings
    # join["wasserstein_distance"] = join.apply(
    #     lambda x: wasserstein_distance(x["vec_item"].numpy(), x["vec_avg_verb"].numpy()), axis=1
    # )  

    # # Calculate the Wasserstein distance between the item and verb embeddings
    # join["wasserstein_distance_10nouns"] = join.apply(
    #     lambda x: wasserstein_distance(x["vec_item"].numpy(), x["vec_avg_verb_10nouns"].numpy()), axis=1
    # )

    # # Calculate the Wasserstein distance between the item and verb embeddings for 50 nouns
    # join["wasserstein_distance_50nouns"] = join.apply(
    #     lambda x: wasserstein_distance(x["vec_item"].numpy(), x["vec_avg_verb_50nouns"].numpy()), axis=1
    # ) 

    # save the dataset with the cosine similarity
    join.drop(columns=["vec_item", "vec_avg_verb", "vec_avg_verb_10nouns", "vec_avg_verb_50nouns"]).to_csv(dataset_path_with_sims, index=False)
    
    # Prepare target variable for all classifiers
    y = join["type"].values
    
    # # ========== NEW CLASSIFIER SECTION: 6 Classifiers ==========
    # print(f"\n{'='*100}")
    # print("NEW CLASSIFIER SECTION: Training 6 Different Feature Sets")
    # print(f"{'='*100}\n")
    
    # all_results = {}
    
    # # CLASSIFIER 1: Concreteness (baseline)
    # X_concreteness = join[['noun_concreteness']].values
    # results_concreteness = train_and_evaluate_classifier(
    #     X_concreteness, y, "Concreteness (noun only)", seeds, label="concreteness"
    # )
    # all_results['concreteness'] = results_concreteness
    
    # # CLASSIFIER 2: SND Metrics
    # snd_cols = ['snd3', 'snd10', 'snd25', 'snd50']
    # X_snd = join[snd_cols].values
    # results_snd = train_and_evaluate_classifier(
    #     X_snd, y, "SND Metrics (snd3, snd10, snd25, snd50)", seeds, label="snd"
    # )
    # all_results['snd'] = results_snd
    
    # # CLASSIFIER 3: Verb-Item Similarity (default context)
    # X_verb_sim = join[['avg_v_sim_to_item']].values
    # results_verb_sim = train_and_evaluate_classifier(
    #     X_verb_sim, y, "Verb-Item Similarity (default context)", seeds, label="verb_sim_default"
    # )
    # all_results['verb_sim_default'] = results_verb_sim
    
    # # CLASSIFIER 4: Verb-Item Similarity (10 nouns context)
    # X_verb_sim_10 = join[['avg_v_sim_to_item_10nouns']].values
    # results_verb_sim_10 = train_and_evaluate_classifier(
    #     X_verb_sim_10, y, "Verb-Item Similarity (10 nouns context)", seeds, label="verb_sim_10nouns"
    # )
    # all_results['verb_sim_10nouns'] = results_verb_sim_10
    
    # # CLASSIFIER 5: Verb-Item Similarity (50 nouns context)
    # X_verb_sim_50 = join[['avg_v_sim_to_item_50nouns']].values
    # results_verb_sim_50 = train_and_evaluate_classifier(
    #     X_verb_sim_50, y, "Verb-Item Similarity (50 nouns context)", seeds, label="verb_sim_50nouns"
    # )
    # all_results['verb_sim_50nouns'] = results_verb_sim_50
    
    # # CLASSIFIER 6: Combined Verb Similarities
    # X_verb_sim_combined = join[['avg_v_sim_to_item', 'avg_v_sim_to_item_10nouns', 'avg_v_sim_to_item_50nouns']].values
    # results_verb_sim_combined = train_and_evaluate_classifier(
    #     X_verb_sim_combined, y, "Verb-Item Similarity (all contexts combined)", seeds, label="verb_sim_combined"
    # )
    # all_results['verb_sim_combined'] = results_verb_sim_combined
    
    # # SUMMARY TABLE
    # print(f"\n{'='*100}")
    # print("SUMMARY: Classifier Performance Across All Feature Sets")
    # print(f"{'='*100}")
    
    # summary_data = []
    # for key, result in all_results.items():
    #     summary_data.append({
    #         'Feature Set': result['feature_name'],
    #         'Mean Accuracy': f"{result['overall_accuracy']:.4f}",
    #         'Std Dev': f"{result['overall_std']:.4f}",
    #         'Full Dataset Accuracy': f"{result['entire_accuracy']:.4f}"
    #     })
    
    # summary_df = pd.DataFrame(summary_data)
    # print(summary_df.to_string(index=False))
    # summary_df.to_csv("data/processed/classifier_summary.csv", index=False)
    # print("\nSummary saved to data/processed/classifier_summary.csv")
    
    # # Save confusion matrices for each classifier
    # for key, result in all_results.items():
    #     cm = confusion_matrix(result['y_true'], result['predictions'])
    #     plt.figure(figsize=(8, 6))
    #     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    #     plt.xticks(ticks=np.arange(len(result['model'].classes_)) + 0.5, labels=result['model'].classes_)
    #     plt.yticks(ticks=np.arange(len(result['model'].classes_)) + 0.5, labels=result['model'].classes_)
    #     plt.xlabel("Predicted")
    #     plt.ylabel("Actual")
    #     plt.title(f"Confusion Matrix - {result['feature_name']}")
    #     plt.tight_layout()
    #     plt.savefig(f"data/processed/confusion_matrix_{key}.png", dpi=300, bbox_inches='tight')
    #     plt.close()
        
    #     # Save model
    #     with open(f"data/processed/svm_model_{key}.pkl", "wb") as f:
    #         pickle.dump(result['model'], f)
    
    # print(f"\nAll 6 classifiers trained and saved successfully!")
    
    # ========== END NEW CLASSIFIER SECTION ==========
    print(f"\n{'='*100}")
    print("BASELINE: Full Experiment Using Embedding Vectors")
    print(f"{'='*100}\n")
    
    #  data for SVM - FULL EXPERIMENT BASELINE (using embeddings)
    X = np.stack(join["vec_item"].values)
 
    print(X.shape)
    # assert X.shape[0] == (200,768)  # 200 items, 768 dimensions

    # Define the parameter grid for GridSearch
    param_grid = {
        'C': [0.1, 1, 10, 100],  # Regularization parameter
        'gamma': ['scale', 'auto'],  # Kernel coefficient
        'kernel': ['linear', 'rbf']  # Kernel type
    }

    # Initialize a list to store the accuracy scores across all seeds
    seed_accuracies = []

    # Loop over seeds
    for seed in seeds:
        print(f"\nRunning with seed {seed}...\n")

        # Define the number of folds for cross-validation
        n_splits = 5
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        # Initialize GridSearchCV with SVM model
        grid_search = GridSearchCV(svm.SVC(), param_grid, cv=kf, scoring='accuracy', n_jobs=-1)

        # Initialize a list to store the accuracy scores for each fold
        accuracy_scores = []

        # Iterate over the folds
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Perform grid search to find the best parameters
            grid_search.fit(X_train, y_train)
      
            # Get the best estimator from the grid search
            best_model = grid_search.best_estimator_

            # Make predictions on the test data using the best model
            y_pred = best_model.predict(X_test)

            # Calculate the accuracy of the classifier for the current fold
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

            # # Print confusion matrix for each fold
            # cm = confusion_matrix(y_test, y_pred)
            # print(f"Confusion matrix for seed {seed}:\n", cm)

        # Calculate the average accuracy for the current seed
        average_accuracy = np.mean(accuracy_scores)
        print(f"Average Accuracy for seed {seed}: {average_accuracy}")
        seed_accuracies.append(average_accuracy)

    # Get the best model from the grid search
    best_model = grid_search.best_estimator_
    # Print the best parameters for the best model
    print(f"Best parameters for seed {seed}: {grid_search.best_params_}")

    # Calculate the overall average accuracy across all seeds
    overall_average_accuracy = np.mean(seed_accuracies)
    print(f"\nOverall Average Accuracy across all seeds: {overall_average_accuracy}")

    # get model f1 precision recall sklearn
    print(classification_report(y_test, y_pred, target_names=best_model.classes_))

    # Retrain the best model on the entire dataset with the final seed for consistency
    final_seed = seeds[-1]
    np.random.seed(final_seed)

    best_model.fit(X, y)
    entire_y_pred = best_model.predict(X)
    entire_accuracy = accuracy_score(y, entire_y_pred)
    print(f"Accuracy on entire dataset with final seed {final_seed}: {entire_accuracy}")

    # Identify misclassified items
    misclassified_mask = (y != entire_y_pred)  # True where predictions are incorrect

    # Filter the misclassified items
    misclassified_items = noun_df[misclassified_mask].copy()

    # Output or save the misclassified items along with their true and predicted labels
    print("Misclassified items:")
    print(misclassified_items[['item', 'type']])  # 'item' is the noun, 'type' is the true label
    print("Predicted labels for misclassified items:")
    print(entire_y_pred[misclassified_mask])  # Predicted labels for the misclassified items

    # Optionally, save the misclassified items to a CSV file
    misclassified_items.loc[:, 'predicted_type'] = entire_y_pred[misclassified_mask]
    # join with join.noun_concreteness
    misclassified_items = misclassified_items.merge(join[['item', 'noun_concreteness']], on='item')
    misclassified_items[['item', 'type', 'predicted_type', 'noun_concreteness']].to_csv(
        f"data/processed/{model_name}_{label}_misclassified_items.csv", index=False
    )

    # Plot confusion matrix
    cm = confusion_matrix(y, entire_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xticks(ticks=np.arange(len(best_model.classes_)) + 0.5, labels=best_model.classes_)
    plt.yticks(ticks=np.arange(len(best_model.classes_)) + 0.5, labels=best_model.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


    # Save the best model and predictions
    with open(f"data/processed/{model_name}_{label}_svm_classifier.pkl", "wb") as f:
        pickle.dump(best_model, f)

    noun_df["prediction"] = entire_y_pred
    noun_df[["item", "type", "prediction"]].to_csv(
        f"data/processed/{model_name}_{label}_predictions.csv", index=False
    )

    # t-SNE plot of embeddings
    tsne = TSNE(n_components=2, random_state=final_seed)
    X_embedded = tsne.fit_transform(X)
    noun_df["tsne_x"] = X_embedded[:, 0]
    noun_df["tsne_y"] = X_embedded[:, 1]
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x="tsne_x", y="tsne_y", hue="type", data=noun_df, palette="tab10")
    plt.title("t-SNE plot of Embeddings")
    plt.show()

    # PCA plot of embeddings
    pca = PCA(n_components=2) 
    X_pca = pca.fit_transform(X)
    noun_df["pca_x"] = X_pca[:, 0]
    noun_df["pca_y"] = X_pca[:, 1]
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x="pca_x", y="pca_y", hue="type", data=noun_df, palette="tab10")
    plt.title("PCA plot of Embeddings")
    plt.show()


# Plot cumulative explained variance
    plt.figure(figsize=(10, 8))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance vs. Number of Components")
    plt.show()

# Plot explained variance ratio
    plt.figure(figsize=(10, 8))
    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Explained Variance Ratio vs. Principal Component")
    plt.show()

if __name__ == "__main__":
    main()
