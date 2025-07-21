import os
import re
import math
import numpy as np
from nltk import word_tokenize
from collections import Counter, defaultdict
from itertools import combinations

# Set paths
input_dir = os.path.join("data", "docs")
output_dir = os.path.join("outputs", "lexrank")
stopwords_file = os.path.join("data", "stopwords.txt")
idf_file = os.path.join("models", "idfs.model.txt")
os.makedirs(output_dir, exist_ok=True)

original_sentences = {}

# Load stopwords
def load_stopwords():
    punctuations = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    stopwords = set()
    with open(stopwords_file, "r") as f:
        for line in f:
            stopwords.add(line.strip())
    stopwords.update(punctuations)
    return stopwords

# Load IDF values
def load_idfs():
    idf = {}
    with open(idf_file, "r") as f:
        for line in f:
            word, val = line.split()
            idf[word] = float(val)
    return idf

# Build similarity matrix using TF-IDF + cosine similarity
def build_similarity_matrix(sentences, stopwords, idfs):
    tfidf_vectors = {}
    global original_sentences
    original_sentences = {}
    
    for i, sent in enumerate(sentences, 1):
        original_sentences[i] = sent
        words = [w.lower() for w in word_tokenize(sent) if w.lower() not in stopwords]
        tf = Counter(words)
        tfidf = {w: (tf[w] * idfs.get(w, 0)) for w in tf}
        tfidf_vectors[i] = tfidf

    cosines = defaultdict(dict)
    for s1, s2 in combinations(tfidf_vectors, 2):
        v1 = tfidf_vectors[s1]
        v2 = tfidf_vectors[s2]
        common = set(v1.keys()) & set(v2.keys())
        if not common:
            sim = 0
        else:
            num = sum(v1[w] * v2[w] for w in common)
            denom = math.sqrt(sum(v ** 2 for v in v1.values())) * math.sqrt(sum(v ** 2 for v in v2.values()))
            sim = num / denom if denom != 0 else 0
        cosines[s1][s2] = sim

    # Make symmetric and fill full matrix
    sentences_ids = list(tfidf_vectors.keys())
    sim_matrix = np.zeros((len(sentences_ids), len(sentences_ids)))
    for i in range(len(sentences_ids)):
        for j in range(len(sentences_ids)):
            if i == j:
                sim_matrix[i][j] = 1.0
            elif sentences_ids[j] in cosines[sentences_ids[i]]:
                sim_matrix[i][j] = cosines[sentences_ids[i]][sentences_ids[j]]
            elif sentences_ids[i] in cosines[sentences_ids[j]]:
                sim_matrix[i][j] = cosines[sentences_ids[j]][sentences_ids[i]]

    return sim_matrix

# LexRank via power method
def power_method(matrix, threshold=0.001, damping=0.85):
    n = matrix.shape[0]
    scores = np.array([1.0 / n] * n)
    while True:
        new_scores = ((1 - damping) / n) + damping * matrix.T.dot(scores)
        if np.linalg.norm(new_scores - scores) < threshold:
            break
        scores = new_scores
    return scores

# Summarize one document
def summarize_lexrank(text, stopwords, idfs, top_k=5):
    sentences = [s.strip() for s in text.split("\n") if s.strip()]
    sim_matrix = build_similarity_matrix(sentences, stopwords, idfs)
    sim_matrix = normalize_rows(sim_matrix)
    scores = power_method(sim_matrix)
    ranked_ids = np.argsort(scores)[::-1][:top_k]
    ranked_ids = sorted(ranked_ids)  # preserve original order
    return [sentences[i] for i in ranked_ids]

#normalise the similarity matrix so that each row sums to 1 
def normalize_rows(matrix):
    for i in range(matrix.shape[0]):
        row_sum = matrix[i].sum()
        if row_sum > 0:
            matrix[i] = matrix[i] / row_sum
    return matrix

# Run for all documents
stopwords = load_stopwords()
idfs = load_idfs()

for idx, filename in enumerate(sorted(os.listdir(input_dir))):
    if not filename.endswith(".txt"):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f"summary_{idx}.txt")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
        summary = summarize_lexrank(text, stopwords, idfs)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in summary:
            f.write(line + "\n")
