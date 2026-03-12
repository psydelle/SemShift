# ---------------------------------------------------------------------------------------------#
# This file contains the code to get the frequency of a word from the SketchEngine API
# and the code to get the word sketch of a word from the SketchEngine API

# ---------------------------------------------------------------------------------------------#


# import the necessary libraries -------------------------------------------------------------#
import functools
import json
import os
import re
import threading
import time
from collections import deque
from typing import List, Tuple

import click
import pandas as pd
import requests as r
import spacy
from dotenv import load_dotenv
from nltk.corpus import wordnet as wn
from tqdm import tqdm

# load the environment variables
load_dotenv()  # take environment variables from .env.

# set the corpus, username, API key
CORPUS = "preloaded/ententen21_tt31"
API_KEY = os.environ.get("SE_API_KEY")
USERNAME = os.environ.get("SE_USERNAME")
BASE_URL = "https://api.sketchengine.eu/bonito/run.cgi"

# Lazy load spaCy model
_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")# , enable=["tokenizer", "tagger", "parser", "lemmatizer", "attribute_ruler", "morphologizer"])
    return _nlp


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
# create a function to get the word sketch of a word and return important data
def get_vn_wordsketch(
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
    d = get_simple_wordsketch(corpus, node, pos="-v", max_items=200)

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

    # time.sleep(1)  # SketchEngine timeout mitigation:(
    # return the dictionary
    return ws


# ---------------------------------------------------------------------------------------------#


class RateLimiter:
    """
    Multi-window sliding rate limiter.

    Enforces:
        - 100 requests per 60 seconds
        - 900 requests per 3600 seconds
        - 2000 requests per 86400 seconds

    Uses timestamp deques to track request history.
    Thread-safe via a single lock.
    """

    def __init__(self):
        # Store timestamps (epoch seconds) of previous calls
        self.minute = deque()
        self.hour = deque()
        self.day = deque()

        # Mutex to ensure thread safety
        self.lock = threading.Lock()

        # (max_requests, window_in_seconds+1)
        self.minute_limit = (100, 61, "minute")
        self.hour_limit = (900, 3601, "hour")
        self.day_limit = (2000, 86401, "day")

    def _wait(self, queue, limit, window, window_name):
        """
        Enforces a single rate window.

        Args:
            queue (deque): Timestamp history for the window
            limit (int): Maximum allowed requests in the window
            window (int): Window duration in seconds
        """
        now = time.time()

        # Remove timestamps that are older than the window
        # These requests no longer count toward the rate limit
        while queue and queue[0] <= now - window:
            queue.popleft()

        # If we are at or above the allowed limit,
        # compute how long until the oldest request expires
        if len(queue) >= limit:
            # Time remaining before the oldest timestamp exits the window
            sleep_for = window - (now - queue[0])

            if sleep_for > 0:
                print(f"Rate limit reached for {window_name}. Sleeping for {sleep_for:.2f} seconds...")
                time.sleep(sleep_for)

    def acquire(self):
        """
        Blocks execution until all rate limits allow a new request.
        Then records the request timestamp in all windows.
        """
        with self.lock:  # Prevent race conditions between threads

            # Enforce each independent time window
            self._wait(self.minute, *self.minute_limit)
            self._wait(self.hour, *self.hour_limit)
            self._wait(self.day, *self.day_limit)

            # Record current request timestamp after waiting
            now = time.time()
            self.minute.append(now)
            self.hour.append(now)
            self.day.append(now)


# Instantiate a global limiter
limiter = RateLimiter()


def _get_request(url, params, retry_backoff=60):
    """
    Function to make a GET request to the SketchEngine API
    Args:
        url (str): url of the API
        params (dict): dictionary of parameters
    Returns:
        response.json() (dict): dictionary with the data from the API
    """
    limiter.acquire()

    try:
        response = r.get(url, params=params, auth=(USERNAME, API_KEY))
        response.raise_for_status()  # Check for any request errors
    except r.exceptions.RequestException as e:
        print("Error occurred during the request:", str(e))
        if response.status_code == 429:
            print(f"Received 429 Too Many Requests. Backing off and retrying in {retry_backoff} seconds.")
            time.sleep(retry_backoff)
            return _get_request(url, params, retry_backoff=retry_backoff*2)  # Retry the request after sleeping
        else:
            raise  # For other types of exceptions, re-raise the error
    return response.json()

# Cache results to avoid redundant API calls. Maxsize is small because consecutive calls are likely to be for the same word.
@functools.lru_cache(maxsize=3)
def get_simple_wordsketch(corpus_name: str, word: str, pos="-v", max_items=200):
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
        "sort_ws_columns": "f",
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
    # TODO: dedupe wrt get_vn_wordsketch, which does the logic but for specific verb and noun
    if verb is None and noun is not None:
        # getting sketch for a noun
        sketch_data = get_simple_wordsketch(corpus_name, noun, pos="-n")  # get noun sketch
        for rel in sketch_data["Gramrels"]:
            if rel["name"] == 'verbs with "%w" as object':
                coll_verbs = rel["Words"]
                colls = [{"verb": x["word"], "noun": noun, "seek": x["seek"]} for x in coll_verbs]
                return colls, sketch_data
    else:
        # getting sketch for a verb (if noun is None), or a specific verb-noun combo (if noun is not None)
        sketch_data = get_simple_wordsketch(corpus_name, verb, pos="-v")  # get verb sketch
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
# @functools.cache # TODO: figure out how not to get kwics twice when querying once by verb and once by noun
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
                verb_lemma=_verb,
                noun_lemma=_noun,
            )
            if not vn_kwics:
                print(f"No KWICs found for {_verb} and {_noun} with seek id {seek_id}")
                continue
            elif len(vn_kwics) < n_kwics:
                print(f"Only got {len(vn_kwics)} KWICs for {_verb} and {_noun} with seek id {seek_id}, expected {n_kwics}")

            # time.sleep(10) # SketchEngine timeout mitigation :(
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
            collocate_position="right",
            verb_lemma=verb,
            noun_lemma=noun,
        )
        verb_noun_pairs = [(verb, noun)] * len(kwics)

    df = pd.DataFrame(
        {"verb": [v for v, n in verb_noun_pairs], "noun": [n for v, n in verb_noun_pairs], "kwics": kwics, "kwic_words": kwic_words}
    )

    return df


def _check_syntax(text_batch, verb_lemma, noun_lemma):
    """
    Check if verb and noun have valid syntactic relationship.
    Only accepts object relationships (direct object, indirect object, passive subject).
    Returns a list of booleans indicating whether each sentence in the batch is valid.
    """
    nlp = get_nlp()
    docs = nlp.pipe(text_batch, disable=["ner"], n_process=-1)

    def filter_doc(doc):
        # Find all verb and noun tokens
        verb_tokens = [t for t in doc if t.lemma_ == verb_lemma and t.pos_ == "VERB"]
        noun_tokens = [t for t in doc if t.lemma_ == noun_lemma and t.pos_ in {"NOUN", "PROPN"}]

        if not verb_tokens or not noun_tokens:
            return False

        # Check each verb-noun pair for valid syntactic relationships
        for verb_tok in verb_tokens:
            for noun_tok in noun_tokens:
                # Skip uppercase nouns
                if not noun_tok.text.islower():
                    continue

                # Check distance constraint
                if abs(noun_tok.i - verb_tok.i) > 10:
                    continue

                # Direct dependency: noun as object of verb
                if noun_tok.head == verb_tok:
                    # Only accept object relations (direct, indirect) and passive subjects
                    if noun_tok.dep_ in {"dobj", "obj", "iobj", "nsubjpass"}:
                        return True

                # Verb in relative clause modifying noun (e.g., "the ball that was kicked")
                # Here the noun is still semantically the object of the verb
                if verb_tok.head == noun_tok:
                    if verb_tok.dep_ in {"relcl", "acl"}:
                        return True

        return False

    return [filter_doc(doc) for doc in docs]


def _get_kwics_from_query(
    corpus_name,
    ws_seek_query=None,
    reg_query=None,
    n_kwics=10,
    n_ctx_sentences=1,
    max_word_count=100,
    collocate_position="right",  # whether to look for the collocate in the right or left context. Should be right for querying verbs and left for querying nouns
    verb_lemma=None,  # for syntactic filtering
    noun_lemma=None,  # for syntactic filtering
    max_pages=20,  # maximum pages to fetch before giving up
):
    is_fallback = ws_seek_query is None
    query = ws_seek_query if not is_fallback else reg_query

    # Enable syntax filtering if both verb and noun lemmas are provided
    use_syntax_filter = verb_lemma is not None and noun_lemma is not None

    def get_data():
        for page in range(1, max_pages + 1):
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
                # get more than we need, to filter out long ones
                "pagesize": n_kwics * 2,
                "kwicleftctx": f"-{n_ctx_sentences+1}:s",  # num sentences of left context
                "kwicrightctx": f"{n_ctx_sentences+1}:s",  # num sentences of right context
            }
            kwics_data = _get_request(BASE_URL + "/concordance", data)
            lines = kwics_data.get("Lines", [])

            if not lines:
                break  # No more results available

            yield lines

    clean_lines = []
    kwic_words = []

    # Tracking statistics
    total_processed = 0
    filtered_syntax = 0
    filtered_other = 0

    for lines in get_data():

        clean_lines_batch = []
        kwic_words_batch = []

        for line in lines:
            total_processed += 1
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
                    filtered_other += 1
                    continue  # skip if collocate not in correct context (inverted order)

                if kwic == kwic2:
                    filtered_other += 1
                    continue  # skip if same word

                if not kwic.isalnum() or not kwic2.isalnum():
                    filtered_other += 1
                    continue  # skip if not alnum

                full_clean = left + [kwic] + right

                if collocate_position == "left":
                    kwic, kwic2 = kwic2, kwic  # swap to maintain verb-noun order
            else:
                raise NotImplementedError("IDK what this does anymore")
                kwic_str = [x["str"] for x in line["Kwic"] if "str" in x]
                if not all([x.isalnum() for x in kwic_str]):
                    filtered_other += 1
                    continue  # skip if not alnum
                kwic, kwic2 = kwic_str[0], kwic_str[-1]
                full_clean = left + kwic_str + right

            if len(full_clean) > max_word_count:
                filtered_other += 1
                continue  # skip if too many words

            expected_kwic_in_left = 1 if collocate_position == "left" else 0

            if left.count(kwic) != expected_kwic_in_left:
                filtered_other += 1
                continue  # more than one instance of the verb appears in the left context

            clean_line = " ".join(full_clean)
            clean_lines_batch.append(clean_line)
            kwic_words_batch.append((kwic, kwic2))

        # Apply syntactic filtering if enabled
        if use_syntax_filter:
            is_valid = _check_syntax(clean_lines_batch, verb_lemma, noun_lemma)
            clean_lines_batch = [line for line, valid in zip(clean_lines_batch, is_valid) if valid]
            kwic_words_batch = [kw for kw, valid in zip(kwic_words_batch, is_valid) if valid]
            filtered_syntax += len(is_valid) - sum(is_valid) # count how many were filtered out by syntax

        clean_lines.extend(clean_lines_batch)
        kwic_words.extend(kwic_words_batch)
        if len(clean_lines) >= n_kwics:
            break

    # Report filtering statistics
    if use_syntax_filter and total_processed > 0:
        print(f"  Filtering stats: {total_processed} processed, {filtered_syntax} rejected by syntax, {filtered_other} rejected by other filters, {len(clean_lines)} kept")

    # keep first N kwics under max_word_count
    if len(clean_lines) > n_kwics:
        clean_lines = clean_lines[:n_kwics]
        kwic_words = kwic_words[:n_kwics]
    elif len(clean_lines) < n_kwics:
        print("  Only", len(clean_lines), "KWICs obtained for", query)
        return None, None

    return clean_lines, kwic_words


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
