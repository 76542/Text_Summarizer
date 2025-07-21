import nltk
import os

# Fix NLTK punkt path
nltk.data.path.append(os.path.join(os.getenv("APPDATA"), "nltk_data"))

import math
import collections
from nltk.tokenize import word_tokenize

# Paths
input_dir = os.path.join("data", "docs")
stopwords_file = os.path.join("data", "stopwords.txt")
output_file = os.path.join("models", "idfs.model.txt")

# Load stopwords
stopwords = set()
with open(stopwords_file, "r") as f:
    for line in f:
        stopwords.add(line.strip())

# Build IDF dictionary
doc_count = 0
word_doc_freq = collections.defaultdict(int) # dict to count how many docs each word appears in 

for filename in sorted(os.listdir(input_dir)):
    if not filename.endswith(".txt"):
        continue

    path = os.path.join(input_dir, filename)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
        tokens = set(word_tokenize(text.lower()))
        tokens = {t for t in tokens if t.isalpha() and t not in stopwords}
        for token in tokens:
            word_doc_freq[token] += 1
        doc_count += 1

# Compute IDF values
idfs = {}
for word, df in word_doc_freq.items():
    idf = math.log(doc_count / (1 + df))  # 1 added for smoothing
    idfs[word] = idf

# Save to file
os.makedirs("models", exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    for word in sorted(idfs.keys()):
        f.write(f"{word} {idfs[word]}\n")

print(f"IDF model written to: {output_file}")
