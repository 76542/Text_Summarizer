import os
from rouge_score import rouge_scorer
from sklearn.metrics import precision_score, recall_score, f1_score
from fuzzywuzzy import fuzz

# Setup paths
summary_dirs = {
    "baseline": os.path.join("outputs", "baseline", "summary_0.txt"),
    "lsa": os.path.join("outputs", "lsa", "summary_0.txt"),
    "lexrank": os.path.join("outputs", "lexrank", "summary_0.txt"),
    "kmedoid": os.path.join("outputs", "kmedoid", "summary_0.txt")
}

reference_path = os.path.join("data", "references", "news_1_reference.txt")
output_file = os.path.join("evaluation", "rouge_scores.txt")
os.makedirs("evaluation", exist_ok=True)

# Load reference
with open(reference_path, "r", encoding="utf-8") as f:
    reference_text = f.read()
    reference_sentences = [line.strip() for line in reference_text.split('\n') if line.strip()]

# Setup scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Capture results
results = ["📊 ROUGE + Precision/Recall/F1 Evaluation Results:\n"]

def compute_prec_rec_f1_fuzzy(ref_sents, sys_sents, threshold=65):
    ref_flags = [0] * len(ref_sents)
    sys_flags = [0] * len(sys_sents)

    print("\n🔍 Checking Sentence Matches with Fuzzy Threshold =", threshold)
    for i, sys_sent in enumerate(sys_sents):
        for j, ref_sent in enumerate(ref_sents):
            ratio = fuzz.token_set_ratio(sys_sent.lower(), ref_sent.lower())
            print(f"\n🟠 SYS: {sys_sent}\n🟢 REF: {ref_sent}\n🔢 Similarity: {ratio}")
            if ratio >= threshold:
                print("✅ MATCHED")
                ref_flags[j] = 1
                sys_flags[i] = 1
                break

    true_pos = sum(sys_flags)
    precision = true_pos / len(sys_sents) if sys_sents else 0.0
    recall = true_pos / len(ref_sents) if ref_sents else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1

# Evaluate each summary
for model, summary_path in summary_dirs.items():
    results.append(f"\n🔹 {model.upper()}")
    if not os.path.exists(summary_path):
        results.append(f"❌ Missing summary for {model}")
        continue

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_text = f.read()
        summary_sentences = [line.strip() for line in summary_text.split('\n') if line.strip()]

    # ROUGE scores
    rouge_scores = scorer.score(reference_text, summary_text)
    results.append(f"  ROUGE-1: {rouge_scores['rouge1'].fmeasure:.4f}")
    results.append(f"  ROUGE-2: {rouge_scores['rouge2'].fmeasure:.4f}")
    results.append(f"  ROUGE-L: {rouge_scores['rougeL'].fmeasure:.4f}")

    # Precision/Recall/F1 based on exact sentence match
    prec, rec, f1 = compute_prec_rec_f1_fuzzy(reference_sentences, summary_sentences)
    results.append(f"  Precision: {prec:.4f}")
    results.append(f"  Recall:    {rec:.4f}")
    results.append(f"  F1-Score:  {f1:.4f}")

# Print and save
for line in results:
    print(line)

with open(output_file, "w", encoding="utf-8") as f:
    for line in results:
        f.write(line + "\n")

print(f"\n✅ Results saved to {output_file}")
