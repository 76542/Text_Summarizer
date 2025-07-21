import numpy as np
import os
import re
from collections import Counter
from scipy.sparse.linalg import svds

# Set paths
input_dir = os.path.join("data", "docs")
output_dir = os.path.join("outputs", "lsa")
os.makedirs(output_dir, exist_ok=True)

# Function to split text into sentences using regex
def split_into_sentences(text):
    pattern = r'.*?\.\s*[A-Z]'
    matches = list(re.finditer(pattern, text, re.DOTALL))
    sentences = []
    for i, match in enumerate(matches):
        start = match.start() if i == 0 else matches[i-1].end()-1
        end = match.end()-1
        sentences.append(text[start:end].strip())
    if matches:
        sentences.append(text[matches[-1].end()-1:].strip())
    return sentences

# LSA-based summarization
def summarize(sentences, top_k=5):
    word_dict = {}
    for sent in sentences:
        for word in sent.lower().split():
            if word not in word_dict:
                word_dict[word] = len(word_dict)

    # Create TF matrix
    M = np.zeros((len(word_dict), len(sentences)))
    for i, sent in enumerate(sentences):
        freqs = Counter(sent.lower().split())
        for word in freqs:
            if word in word_dict:
                M[word_dict[word]][i] = freqs[word]

    # Apply SVD
    try:
        U, S, VT = svds(M, k=min(5, len(sentences)-1))
    except:
        return sentences[:top_k]  # fallback if SVD fails

    # Score sentences
    sentence_scores = np.dot(S, VT)
    ranked = [(score, i) for i, score in enumerate(sentence_scores)]
    ranked = sorted(ranked, key=lambda x: -x[0])[:top_k]
    ranked = sorted(ranked, key=lambda x: x[1])  # keep original order

    return [sentences[i] for _, i in ranked]

# Run on all input documents
for index, filename in enumerate(sorted(os.listdir(input_dir))):
    if not filename.endswith(".txt") or filename.startswith("."):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f"summary_{index}.txt")

    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
        sentences = split_into_sentences(raw_text)
        top_k = max(1, int(0.2 * len(sentences)))  # 20% of total sentences, minimum 1
        summary = summarize(sentences, top_k=top_k)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in summary:
            f.write(line.strip() + "\n")
