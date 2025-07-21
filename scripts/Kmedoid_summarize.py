import os
import random
import itertools
import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem import LancasterStemmer

# Setup
input_file = os.path.join("data", "docs", "news_1.txt")
output_file = os.path.join("outputs", "kmedoid", "summary_0.txt")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Read input sentences
with open(input_file, "r", encoding="utf-8") as f:
    raw = f.read()
    sentence_list = [s.strip() for s in raw.split("\n") if s.strip()]

# Clean and preprocess
def preprocess(sentences):
    stemmer = LancasterStemmer()
    processed = {}
    for idx, sentence in enumerate(sentences):
        tokens = word_tokenize(sentence.lower())
        tags = pos_tag(tokens)
        # keep only nouns
        nouns = [stemmer.stem(w) for w, tag in tags if tag.startswith("NN")]
        processed[idx] = nouns
    return processed

dict_sentence = preprocess(sentence_list)

# Jaccard Similarity between noun sets
def jaccard_sim(a, b):
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

# Build pairwise similarity
sentence_pairs = list(itertools.combinations(dict_sentence.keys(), 2))
distance_matrix = {}
for a, b in sentence_pairs:
    sim = jaccard_sim(dict_sentence[a], dict_sentence[b])
    distance_matrix[(a, b)] = 1 - sim

# KMedoid clustering
def kmedoid_clustering(k=3, max_iter=100):
    medoids = random.sample(list(dict_sentence.keys()), k)
    for _ in range(max_iter):
        clusters = {m: [] for m in medoids}
        for i in dict_sentence.keys():
            best_medoid = min(medoids, key=lambda m: distance_matrix.get(tuple(sorted((m, i))), 1))
            clusters[best_medoid].append(i)
        new_medoids = []
        for cluster_points in clusters.values():
            min_sum = float("inf")
            best_point = None
            for i in cluster_points:
                total = sum(distance_matrix.get(tuple(sorted((i, j))), 1) for j in cluster_points if i != j)
                if total < min_sum:
                    min_sum = total
                    best_point = i
            new_medoids.append(best_point)
        if set(new_medoids) == set(medoids):
            break
        medoids = new_medoids
    return medoids

# Run clustering
summary_indices = kmedoid_clustering(k=3)
summary_indices.sort()

# Save summary
with open(output_file, "w", encoding="utf-8") as f:
    for i in summary_indices:
        f.write(sentence_list[i] + "\n")

print("✅ KMedoid summary written to:", output_file)
