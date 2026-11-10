from nltk.corpus import words
import string
from tqdm import tqdm
import json
import argparse
import random
import numpy as np
import copy

from pathlib import Path
ROOT = Path(__file__).parent
SCORES_FILE = ROOT / "word_scores.json"

def get_words():
    """Return set of standardised 5-letter words"""
    wordle_words = set([w.lower() for w in words.words() if (len(w) == 5)])
    return wordle_words

def non_repetitive_spellings(wordle_list):
    """Return all 5 letter words that have 5 unique letters"""
    nrs = set()
    for w in wordle_list:
        if len(set(w)) == 5:
            nrs.add(w)
    return nrs

def get_statistics(wordle_words):
    """For each word, add 1 to the correct position where a letter was discovered"""

    alphabet = string.ascii_lowercase
    letter_counts = {alph: [0,0,0,0,0] for alph in alphabet}

    # print(f"DEBUG 1: Letter counts at init\n{letter_counts}")
    num_words = len(wordle_words)

    for word in tqdm(wordle_words, desc="5-letter words", total=num_words):
        for i, alph in enumerate(word):
            char_ohe = letter_counts[alph]
            char_ohe[i] += 1
            letter_counts[alph] = char_ohe
            # print(f"DEBUG 2: Char {alph} in Word {word} --- new_ohe {letter_counts[alph]}")
    
    return letter_counts

def get_totals(char_stats):
    """Compile results for each letter: total occurrences, and most frequent position."""
    char_totals = {}
    for alph, stats in char_stats.items():
        char_totals[alph] = sum(stats)
    return char_totals

def compute_alphabet_ranking(wordle_words):
    # for each letter of the alphabet, store it's total and positional appearances
    # test_words = ["acted", "movie", "weird", "about", "hymns", "clone", "first", "place", "crash", "water", "quill", "looks", "given"]
    char_stats_pos = get_statistics(wordle_words)
    
    # The best words to start are those with the most frequent letters in their most frequent positions
    char_totals = get_totals(char_stats_pos)
    # json_totals = json.dumps(char_totals, indent=2)
    # print(json_totals)

    # Rank each letter, give it a weight for it's rank
    ranked_alphabet = dict(
        sorted(char_totals.items(), key=lambda item: item[1], reverse=True)
    )
    for i, (k, v) in enumerate(ranked_alphabet.items()):
        ranked_alphabet[k] = 1/(i+1)
    print(ranked_alphabet)

    # normalise positions for each char so that the list sums to 1
    char_stats_pos_norm = dict()
    for k, v in char_stats_pos.items():
        char_stats_pos_norm[k] = [float(pos_count)/char_totals[k] for pos_count in v]

    return ranked_alphabet, char_stats_pos_norm

def word_score(w, rank, pos_w):
    """Return a score for each word sum(rank*i_pos_weight)_i"""
    score = 0
    for i, alph in enumerate(w):
        i_pos_weight = pos_w[alph][i]
        score += (rank[alph] * i_pos_weight)
    return score

def best_words(wordle_words, rank, pos_w):
    nr_words = non_repetitive_spellings(wordle_words)
    total_candidates = len(nr_words)
    word_scores = dict()
    for w in tqdm(nr_words, desc="Scoring candidate words", total=total_candidates):
        # compute a score for each word
        word_scores[w] = word_score(w, rank, pos_w)

    words_sorted_by_score = dict(
        sorted(word_scores.items(), key=lambda item: item[1], reverse=True)
    )
    return words_sorted_by_score

def best_starters(scored_words, min_unique=12):
    """
    Pick w1, w2, and w3 randomly so that they have at least `min_unique` unique letters between them.
    Then remove the word with the lowest score and find an alternative, always making sure to have `min_unique` values in the curr_set.
        Loop until 
    """
    curr_set = set()
    chosen_w = list()
    all_words = list(scored_words)
    while len(curr_set) < min_unique:
        chosen_w = random.sample(all_words, k=3)
        curr_set = set(chosen_w[0] + chosen_w[1] + chosen_w[2])

    chosen_w_scores = [scored_words[w] for w in chosen_w]
    curr_score = sum(chosen_w_scores)
    print(f"Starting set:\nWords = {chosen_w}\nSet = {curr_set}\nScores = {chosen_w_scores}\nTotal = {curr_score}")

    iterate = True
    while iterate:
        lowest_scoring_word_idx = np.argmin(chosen_w_scores)
        removed_word_score = chosen_w_scores[lowest_scoring_word_idx]
        temp_chosen_w = copy.deepcopy(chosen_w)
        temp_chosen_w.pop(lowest_scoring_word_idx)
        temp_chosen_scores = copy.deepcopy(chosen_w_scores)
        temp_chosen_scores.pop(lowest_scoring_word_idx)

        curr_set = set(temp_chosen_w[0] + temp_chosen_w[1])

        valid_replacements = dict()
        for w in all_words:
            if len(curr_set | set(w)) >= min_unique:
                valid_replacements[w] = scored_words[w]
        # get the highest scoring valid word
        highest_scorer = max(valid_replacements, key=valid_replacements.get)
        temp_chosen_w.append(highest_scorer)
        temp_chosen_scores.append(scored_words[highest_scorer])

        if scored_words[highest_scorer] > removed_word_score:
            chosen_w = temp_chosen_w
            chosen_w_scores = temp_chosen_scores
            curr_set = set(chosen_w[0] + chosen_w[1] + chosen_w[2])

        else: 
            iterate = False

    return chosen_w, chosen_w_scores, sum(chosen_w_scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--solve", action="store_true")
    args = parser.parse_args()
    if args.score:
        # get all 5 letter words in english
        wordle_words = get_words()

        # get the frequency rank for each letter, and a weight-vector indicating how frequently each letter appears in a given position
        rank, pos_weights = compute_alphabet_ranking(wordle_words)

        # Finally, find 3 words which use exactly the 15 most common letters, and also each letter's most common position where possible
        word_scores = best_words(wordle_words, rank, pos_weights)

        with open('word_scores.json', 'w', encoding='utf-8') as f:
            json.dump(word_scores, f, indent=2)

    if args.solve:
        if SCORES_FILE.exists():
            with open(SCORES_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            chosen_w, scores, total_score = best_starters(data)
            # get three words with 15 unique letters among them which have the highest possible combined score
        else:
            print("Scores need to be calculated first.")