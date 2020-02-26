import numpy as np
import re
import multiprocessing
import multiprocessing as mp
from janome.tokenizer import Tokenizer
import os
import pickle


current_path = os.path.dirname(os.path.abspath(__file__))

MAX_LENGTH = 30
EN_PATH = os.path.join(current_path, 'data', 'kyoto-dev.en')
JA_PATH = os.path.join(current_path, 'data', 'kyoto-dev.ja')

# output files
EN_LANG_PATH = os.path.join(current_path, 'data', 'normalized', 'normalized.en')
JA_LANG_PATH = os.path.join(current_path, 'data', 'normalized', 'normalized.ja')

def normalize_en(s):
    """ Processes an English string by removing non-alphabetical characters (besides .!?).
    """
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^\w.!?]+", r" ", s, flags=re.UNICODE)
    return s


def normalize_ja(s, segmenter):
    """ Processes a Japanese string by removing non-word characters and separating tokens with spaces.
    """
    s = s.strip()
    s = re.sub(r"[^\w.!?。]+", r" ", s, flags=re.UNICODE)
    tokenized_list = []
    for token in segmenter.tokenize(s):
        tokenized_list.append(token.base_form)

    s = " ".join(tokenized_list)
    s = re.sub("\s+", " ", s).strip()
    return s

def normalize(en_lines, ja_lines):
    """ Process lists of both English and Japanese strings.
    """
    tokenizer = Tokenizer()
    return [[normalize_en(l1), normalize_ja(l2, tokenizer)] for l1, l2 in zip(en_lines, ja_lines)]

def read_langs(en_file, ja_file):
    """ Reads corpuses and returns a Lang object for each language and all normalized sentence pairs.
    """
    en_lines = open(en_file, encoding="utf8", errors="ignore").read().split("\n")
    ja_lines = open(ja_file, encoding="utf8", errors="ignore").read().split("\n")

    n_processes = multiprocessing.cpu_count() - 1
    pool = mp.Pool(n_processes)
    interval = len(en_lines) // n_processes
    results = [
        pool.apply_async(
            normalize, args=(en_lines[i * interval:(i + 1) * interval], ja_lines[i * interval:(i + 1) * interval]))
        for i in range(n_processes)
    ]
    pairs = []
    for p in results:
        pairs += p.get()

    return pairs

def filter_pair_by_len(p, maxlen=MAX_LENGTH):
    """ Filter out sentences if they are greater than maximum length.
    """
    return len(p[0].split(" ")) < maxlen and len(p[1].split(" ")) < maxlen


if __name__ == "__main__":
    en = []
    ja = []
    pairs = read_langs(EN_PATH, JA_PATH)
    print("Number of sentences:", len(pairs))
    pairs = [pair for pair in pairs if filter_pair_by_len(pair, MAX_LENGTH)]
    print("Number of trimmed sentences:", len(pairs))

    for pair in pairs:
        en.append(pair[0])
        ja.append(pair[1])

    np.savetxt(EN_LANG_PATH, np.array(en), fmt="%s", encoding='utf-8')
    np.savetxt(JA_LANG_PATH, np.array(ja), fmt="%s", encoding='utf-8')

    # pickle.dump(en, open(EN_LANG_PATH, "wb"))
    # pickle.dump(ja, open(JA_LANG_PATH, "wb"))