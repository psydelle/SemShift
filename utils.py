# ---------------------------------------------------------------------------------------------#
# This file contains the code to get the frequency of a word from the SketchEngine API
# and the code to get the word sketch of a word from the SketchEngine API

# ---------------------------------------------------------------------------------------------#


# import the necessary libraries -------------------------------------------------------------#
import re
from typing import List, Tuple
from nltk.corpus import wordnet as wn
import pandas as pd
import json
from tqdm import tqdm
import click
import os
import time
from dotenv import load_dotenv
import requests as r
import functools


# load the environment variables
load_dotenv()  # take environment variables from .env.

# set the corpus, username, API key
CORPUS = "preloaded/ententen21_tt31"
API_KEY = os.environ.get("SE_API_KEY")
USERNAME = os.environ.get("SE_USERNAME")
BASE_URL = "https://api.sketchengine.eu/bonito/run.cgi"


# ---------------------------------------------------------------------------------------------#

# Data Processing Functions -------------------------------------------------------------------#


def clean_sentence(sentence: str) -> str:
    """
    Clean a sentence by performing the following operations:
    - Convert to lowercase
    - Remove URLs
    - Remove HTML tags
    - Remove punctuation
    - Remove extra whitespace
    - Remove digits (optional, depending on your use case)

    Args:
        sentence (str): The sentence to be cleaned.

    Returns:
        str: The cleaned sentence.
    """
    # Convert to lowercase
    sentence = sentence.lower()

    # Remove URLs
    sentence = re.sub(r"http\S+|www\S+|https\S+", "", sentence, flags=re.MULTILINE)

    # # Remove HTML tags
    # sentence = re.sub(r'<.*?>', '', sentence)

    # # Remove punctuation
    # sentence = sentence.translate(str.maketrans('', '', string.punctuation))

    # Remove digits (optional)
    # sentence = re.sub(r'\d+', '', sentence)

    # Remove extra whitespace
    sentence = " ".join(sentence.split())

    return sentence


# Functions to get Sketch Engine Data --------------------------------------------------------#
@functools.cache
def get_word_freq(
    word: str,
    pos: str,
    corpus=CORPUS,
    base_url=BASE_URL,
    username=USERNAME,  # set your username
    api_key=API_KEY,  # set your API key, you can find it in your profile on SketchEngine
):
    """Function to get the frequency of a word from the SketchEngine API
    Args:
        word (str): word to get the frequency of
        pos (str): part of speech tag of the word
        corpus (str): name of the corpus
        base_url (str): base url of the SketchEngine API
        USERNAME (str): username
        API_KEY (str): API key
        n_gram (bool): whether we want to get the frequency of an n-gram or a unigram
    Returns:
        freq (int): frequency of the word
    """
    data = {
        "corpname": corpus,
        "format": "json",
        "lemma": word,
        "lpos": pos,
    }
    # get the data from the API
    d = r.get(
        base_url + "/wsketch?corpname=%s" % data["corpname"],
        params=data,
        auth=(username, api_key),
    ).json()
    # get the frequency
    freq = d["freq"]
    # print the frequency
    print(f"The frequency of {word} as a {pos} is {freq}")

    time.sleep(4)  # SketchEngine timeout mitigation :(
    # return the frequency
    return freq


# create a function to get the word sketch of a word and return important data
def get_word_sketch(
    node: str,  # for our experiment, this is the verb
    collocate: str,  # for our experiment, this is the noun (Direct Object)
    corpus=CORPUS,
    base_url=BASE_URL,
    username=USERNAME,
    api_key=API_KEY,
):
    """Function to get the word sketch of a word from the SketchEngine API
    Args:
        node (str): node of the collocation
        collocate (str): collocate of the collocation
        corpus (str): name of the corpus
        base_url (str): base url of the SketchEngine API
        USERNAME (str): SketchEngine username
        API_KEY (str): SketchEngine API key
    Returns:
        ws (dict): dictionary with the word sketch data
    """
    data = {
        "corpname": corpus,
        "format": "json",
        "lemma": node,
        "lpos": "-v",
        "sort_gramrels": 1,
        "maxitems": 200,
        "sort_ws_columns": "f",
        # "expand_seppage": 1,
    }
    # get the data from the API
    d = r.get(
        base_url + "/wsketch",
        params=data,
        auth=(username, api_key),
    ).json()

    # create an empty dictionary to store the wordsketch data
    ws = {}
    # get the gramrels
    gramrels = d["Gramrels"]
    for entry in gramrels:
        # print(list(entry.keys()))
        if entry["name"] == 'objects of "%w"':
            obj_entry = entry
            break  # stop the loop when we find the entry we want

    # loop through the collocates
    for entry in obj_entry["Words"]:
        # find the collocate we want
        if entry["word"] == collocate:
            # get the data
            ws["Verb"] = node
            ws["Noun"] = entry["word"]
            ws["CollFreq"] = entry["count"]
            ws["Score"] = entry["score"]
            ws["Usage"] = entry["cm"]
            # get tt key values from High
            ws["Context"] = entry["High"]
            break  # stop the loop when we find the entry we want
        # if we cannot find the collocate,
        # set the values to None and we'll deal with it manually
        else:
            ws["Verb"] = node
            ws["Noun"] = collocate
            ws["CollFreq"] = "Not found"
            ws["Score"] = "Not found"
            ws["Usage"] = "Not found"
            ws["Context"] = "Not found"

    time.sleep(1)  # SketchEngine timeout mitigation:(
    # return the dictionary
    return ws


def get_ws_data(verb, noun, data: str = "coll_freq"):
    """
    Function to get the collocation frequency of a verb and noun from the SketchEngine API
    Args:
        verb (str): verb to get the collocation frequency of
        noun (str): noun to get the collocation frequency of
    Returns:
        coll_data["freq"] (int): frequency of the collocation
    """

    coll_data = get_word_sketch(verb, noun)
    if data == "coll_freq":
        return coll_data["CollFreq"]
    elif data == "score":
        return coll_data["Score"]
    elif data == "usage":
        return coll_data["Usage"]
    elif data == "context":
        return coll_data["Context"]
    else:
        raise ValueError(
            "Invalid data type. Please choose from 'coll_freq', 'score', 'usage', or 'context'."
        )


# ---------------------------------------------------------------------------------------------#


def _get_request(url, params):
    """
    Function to make a GET request to the SketchEngine API
    Args:
        url (str): url of the API
        params (dict): dictionary of parameters
    Returns:
        response.json() (dict): dictionary with the data from the API
    """
    try:
        response = r.get(url, params=params, auth=(USERNAME, API_KEY))
        response.raise_for_status()  # Check for any request errors
    except r.exceptions.RequestException as e:
        print("Error occurred during the request:", str(e))
    return response.json()


def get_ws(corpus_name: str, word: str, pos="-v", max_items=200):
    """
    Function to get the word sketch of a word from the SketchEngine API
    Args:
        corpus_name (str): name of the corpus
        word (str): word to get the word sketch of
        pos (str): part of speech tag of the word
        max_items (int): maximum number of items to return
    Returns:
        sketch_data (dict): dictionary with the word sketch data
    """
    data = {
        "corpname": corpus_name,
        "format": "json",
        "lemma": word,
        "lpos": pos,
        "maxitems": max_items,
        "structured": 1,
        "minfreq": 1,
    }
    sketch_data = _get_request(BASE_URL + "/wsketch?corpname=%s" % data["corpname"], data)
    return sketch_data


def get_verb_noun_sketch_seek_id(corpus_name, verb, noun) -> Tuple[List[dict], dict]:
    """
    Function to get the seek id of a verb and noun from the word sketch
    Args:
        corpus_name (str): name of the corpus
        verb (str): verb to get the seek id of. If None, return seek ids for all verbs with the noun as object
        noun (str): noun to get the seek id of. If None, return seek ids for all object nouns
    Returns:
        colls (List[dict]): list of dictionaries with the verb, noun, and seek id
        sketch_data (dict): dictionary with the word sketch data
    """
    # get the wordsketch seek id
    if verb is None and noun is not None:
        # getting sketch for a noun
        sketch_data = get_ws(corpus_name, noun, pos="-n")  # get noun sketch
        for rel in sketch_data["Gramrels"]:
            if rel["name"] == 'verbs with "%w" as object':
                coll_verbs = rel["Words"]
                colls = [{"verb": x["word"], "noun": noun, "seek": x["seek"]} for x in coll_verbs]
                return colls, sketch_data
    else:
        # getting sketch for a verb (if noun is None), or a specific verb-noun combo (if noun is not None)
        sketch_data = get_ws(corpus_name, verb, pos="-v")  # get verb sketch
        for rel in sketch_data["Gramrels"]:
            if rel["name"] == 'objects of "%w"':  # get the objects of the verb
                coll_nouns = rel["Words"]
                colls = [{"verb": verb, "noun": x["word"], "seek": x["seek"]} for x in coll_nouns]
                if noun is None:
                    # if no noun is specified, return seek ids for all object nouns
                    return colls, sketch_data
                else:
                    # get the seek id of the noun we want
                    return [x for x in colls if x["noun"] == noun], sketch_data
    raise ValueError(f"Could not find seek id for {verb} and {noun}")


def get_corp_info(corpus_name):
    """
    Function to get the information of a corpus from the SketchEngine API
    Args:
        corpus_name (str): name of the corpus
    Returns:
        corp_data (dict): dictionary with the corpus data
    """

    data = {"corpname": corpus_name}
    corp_data = _get_request("https://api.sketchengine.eu/search/corp_info", data)
    return corp_data


# For KWICS from Sketch Engine -----------------------------------------------------------------#
def get_vn_kwics(
    corpus_name,
    verb=None,
    noun=None,
    # wordsketch_seek_id=None,
    n_kwics=10,
    max_word_count=100,
    n_ctx_sentences=1,
    subset_to_top_n_pairs=None,  # by default, get all pairs
):
    try:
        assert verb is not None or noun is not None, "Must specify at least a verb or a noun to get KWICs"
        assert subset_to_top_n_pairs is None or (verb is None or noun is None), "Subset to top N pairs only works when specifying either verb or noun, not both"

        if subset_to_top_n_pairs is None:
            subset_to_top_n_pairs = 99999999

        # if wordsketch_seek_id is None:
        vn_results, _sketch_data = get_verb_noun_sketch_seek_id(
            corpus_name, verb, noun
        )

        # get top N vn pairs. vn_results is a list of dicts with verb, noun, and seek id
        kwics = []
        kwic_words = []
        verb_noun_pairs = []
        i = 0
        for vn in vn_results:
            _verb = vn["verb"]
            _noun = vn["noun"]
            seek_id = vn["seek"]
            # filter when noun is not all lowercase, to avoid garbage e.g., Framed Print
            if not _noun.islower():
                continue
            ws_seek_query = f"w{seek_id} within <s />"
            vn_kwics, vn_kwic_words = _get_kwics_from_query(
                corpus_name,
                ws_seek_query=ws_seek_query,
                n_kwics=n_kwics,
                n_ctx_sentences=n_ctx_sentences,
                max_word_count=max_word_count,
                collocate_position="right" if verb is not None else "left",  # if we're querying by verb, look for noun in right context, and vice versa
            )
            if not vn_kwics:
                continue
            time.sleep(4) # SketchEngine timeout mitigation :(
            kwics.extend(vn_kwics)
            kwic_words.extend(vn_kwic_words)
            verb_noun_pairs.extend([(_verb, _noun)] * len(vn_kwics))
            print(f"Got {len(vn_kwics)} KWICs for {_verb} and {_noun}")
            i += 1
            if i == subset_to_top_n_pairs:
                break
        if i < subset_to_top_n_pairs:
            print(f"Only got {i} entries for verb: {verb} and noun: {noun} before running out of pairs")

    except ValueError:
        print(f"Could not find seek id for {verb} and {noun}, falling back to default")
        if noun is None or verb is None:
            raise ValueError("Cannot get KWICs for all verbs/nouns without seek ids")
        query = f'q[lempos_lc="{verb}-v"][]?[lempos_lc="{noun}-n"] within <s />'
        kwics, kwic_words = _get_kwics_from_query(
            corpus_name,
            reg_query=query,
            n_kwics=n_kwics,
            n_ctx_sentences=n_ctx_sentences,
            max_word_count=max_word_count,
            collocate_position="right"
        )
        verb_noun_pairs = [(verb, noun)] * len(kwics)

    df = pd.DataFrame(
        {"verb": [v for v, n in verb_noun_pairs], "noun": [n for v, n in verb_noun_pairs], "kwics": kwics, "kwic_words": kwic_words}
    )

    return df


def _get_kwics_from_query(
    corpus_name,
    ws_seek_query=None,
    reg_query=None,
    n_kwics=10,
    n_ctx_sentences=1,
    max_word_count=100,
    collocate_position="right",  # whether to look for the collocate in the right or left context. Should be right for querying verbs and left for querying nouns
):
    is_fallback = ws_seek_query is None
    query = ws_seek_query if not is_fallback else reg_query

    def get_data():
        for page in range(1, 3):
            data = {
                "corpname": corpus_name,
                "q": query,
                "concordance_query[queryselector]": "iqueryrow",
                "concordance_query[iquery]": query,
                "default_attr": "lemma",
                "attr": "word",
                # "refs": "=bncdoc.alltyp",
                "attr_allpos": "all",
                "cup_hl": "q",
                "structs": "s,g",
                "fromp": page,
                # get more than we need, to filter out long ones. TODO: requery on demand
                "pagesize": n_kwics * 2,
                "kwicleftctx": f"-{n_ctx_sentences+1}:s",  # num sentences of left context
                "kwicrightctx": f"{n_ctx_sentences+1}:s",  # num sentences of right context
            }
            kwics_data = _get_request(BASE_URL + "/concordance", data)
            lines = kwics_data["Lines"]

            yield from lines

    clean_lines = []
    kwic_words = []
    for line in get_data():
        # filter "strc", like sentence breaks
        left = [x["str"] for x in line["Left"] if "str" in x]
        right = [x["str"] for x in line["Right"] if "str" in x]

        if not is_fallback:
            assert len(line["Kwic"]) == 1, f"Expected only one kwic, got {line['Kwic']}"
            kwic = line["Kwic"][0]["str"]
            # grab collocate from context corresponding to collocate_position
            # it will be the first element which has the "coll": 1 key
            _coll_context = line["Right"] if collocate_position == "right" else line["Left"]
            kwic2 = next((x["str"] for x in _coll_context if "coll" in x), None)
            
            if kwic2 is None:
                continue  # skip if collocate not in correct context (inverted order)

            if kwic == kwic2:
                # skip if same word
                continue

            if not kwic.isalnum() or not kwic2.isalnum():
                # skip if not alnum
                continue

            full_clean = left + [kwic] + right

            if collocate_position == "left":
                kwic, kwic2 = kwic2, kwic  # swap to maintain verb-noun order
        else:
            raise NotImplementedError("IDK what this does anymore")
            kwic_str = [x["str"] for x in line["Kwic"] if "str" in x]
            if not all([x.isalnum() for x in kwic_str]):
                # skip if not alnum
                continue
            kwic, kwic2 = kwic_str[0], kwic_str[-1]
            full_clean = left + kwic_str + right

        if len(full_clean) > max_word_count:
            # skip if too many words
            continue

        expected_kwic_in_left = 1 if collocate_position == "left" else 0

        if left.count(kwic) != expected_kwic_in_left:
            # more than one instance of the verb appears in the left context
            # filter to avoid doing character vs. token offset math later
            continue

        clean_line = " ".join(full_clean)
        clean_lines.append(clean_line)
        kwic_words.append((kwic, kwic2))
        if len(clean_lines) >= n_kwics:
            break

    # keep first N kwics under max_word_count
    if len(clean_lines) != n_kwics:
        print("Only", len(clean_lines), "KWICs obtained for", query)
        return None, None

    return clean_lines, kwic_words


# def process_item(line):
#     """
#     Function to process an item from the dataframe
#     Args:
#         line (dict): dictionary with the data from the dataframe
#     Returns:
#         data (dict): dictionary with the processed data
#     """
#     data = {}
#     kwics_sentences, kwic_words = get_vn_kwics(
#         CORPUS,
#         line["verb"],
#         line["noun"],
#         n_kwics=100,
#         n_ctx_sentences=1,
#         max_word_count=100,
#     )
#     data["verb"] = line["verb"]
#     data["noun"] = line["noun"]
#     data["kwics"], data["kwic_words"] = kwics_sentences, kwic_words
#     return data


# def process_line(line):
#     ''' Function to process a line from the dataframe

#     Args:
#         line (dict): dictionary with the data from the dataframe
#     Returns:
#         p (dict): dictionary with the processed data

#     '''
#     p = process_item(line)
#     if not p["kwics"]:
#         print("No KWICS found for", line["item"])
#     time.sleep(1)  # SketchEngine timeout mitigation:(
#     return (line["item"], p)

# #  # decorator to create a command line interface
# # @click.command()
# # @click.option("-f", "--csv-file", required=True)
# # @click.option("-o", "--out-file", required=True)


# def process_corpus(csv_file, out_file):
#     ''' Function to process the corpus and save the data to a json file
#     Args:
#         csv_file (str): path to the csv file
#         out_file (str): path to the output file
#     '''
#     in_df = pd.read_csv(csv_file)

#     data = [process_line(line) for line in tqdm(in_df.iloc, total=len(in_df))]

#     data = dict(data)

#     with open(out_file, "w") as f:
#         json.dump(data, f, ensure_ascii=True)


# # if __name__ == "__main__":
# #     process_corpus()


# ---------------------------------------------------------------------------------------------#


# Getting info from WordNet ------------------------------------------------------------------#


def get_len_synset(word: str, pos=wn.VERB):
    """
    Function to get the length of the synsets of a verb
    Args:
        word (str): word to get the synsets of
        pos (str): part of speech tag of the word from WordNet
    Returns:
        len(synsets) (int): number of synsets of the word
    """
    synsets = wn.synsets(word, pos=pos)  # returns a list of synsets for a word
    return len(synsets)


# ----------------------------------------------------------------------------------------------#

# Get concreteness ratings (Brysbaert et al., 2014) -------------------------------------------#


def get_concreteness_rating(word: str):
    """
    Function to get the concreteness rating of a word from the Brysbaert et al. (2014) dataset
    Args:
        df (pd.DataFrame): dataframe with the concreteness ratings
        col_name (str): name of the column with the concreteness ratings
    Returns:
        df (pd.DataFrame): dataframe with the concreteness ratings
    """
    # read the data
    concreteness = pd.read_csv("data/concreteness_ratings.csv")
    # get the concreteness rating
    if word in concreteness["Word"].values:
        rating = concreteness[concreteness["Word"] == word]["Conc.M"].values[0]
    else:
        rating = None

    return rating


# Semantic Neighbourhood Density Data --------------------------------------------------#
# For nouns only. From Reilly & Desai (2017) dataset

def load_snd_data():
    """Load SND metrics from file (word, PoS, snd3, snd10, snd25, snd50)"""
    try:
        snd_df = pd.read_csv("data/snd_reilly_desai.txt", sep=r'\s+', engine='python')
        snd_df.columns = ['word', 'PoS', 'snd3', 'snd10', 'snd25', 'snd50']
        # Filter for nouns only
        snd_df = snd_df[snd_df['PoS'] == 'NN'].copy()
        snd_dict = {}
        for _, row in snd_df.iterrows():
            word = row['word']
            # Skip rows where word is NaN or not a string
            if pd.isna(word) or not isinstance(word, str):
                continue
            snd_dict[word.lower()] = {
                'snd3': row['snd3'],
                'snd10': row['snd10'],
                'snd25': row['snd25'],
                'snd50': row['snd50']
            }
        return snd_dict
    except FileNotFoundError:
        print("Warning: snd_reilly_desai.txt not found")
        return {}
