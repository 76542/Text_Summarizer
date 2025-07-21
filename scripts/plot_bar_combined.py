import os
import matplotlib.pyplot as plt

# Load scores from file
results_file = os.path.join("evaluation", "rouge_scores.txt")
scores = {
    "baseline": {},
    "lsa": {},
    "lexrank": {},
    "kmedoid": {}
}

# Parse the score file safely
current_model = None
with open(results_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith("🔹"):
            current_model = line[2:].strip().lower()
            if current_model not in scores:
                scores[current_model] = {}
        elif current_model:
            try:
                if "ROUGE-1" in line:
                    scores[current_model]["rouge1"] = float(line.split(":")[1].strip())
                elif "ROUGE-2" in line:
                    scores[current_model]["rouge2"] = float(line.split(":")[1].strip())
                elif "ROUGE-L" in line:
                    scores[current_model]["rougeL"] = float(line.split(":")[1].strip())
                elif "Precision" in line:
                    scores[current_model]["precision"] = float(line.split(":")[1].strip())
                elif "Recall" in line:
                    scores[current_model]["recall"] = float(line.split(":")[1].strip())
                elif "F1-Score" in line:
                    scores[current_model]["f1"] = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                key = line.split(":")[0].strip().lower()
                scores[current_model][key] = 0.0

# Prepare data for plotting
models = list(scores.keys())
rouge1 = [scores[m].get("rouge1", 0.0) for m in models]
rouge2 = [scores[m].get("rouge2", 0.0) for m in models]
rougeL = [scores[m].get("rougeL", 0.0) for m in models]

precision = [scores[m].get("precision", 0.0) for m in models]
recall = [scores[m].get("recall", 0.0) for m in models]
f1 = [scores[m].get("f1", 0.0) for m in models]

x = range(len(models))
width = 0.25

# Create side-by-side bar plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROUGE subplot
axes[0].bar([i - width for i in x], rouge1, width=width, label="ROUGE-1")
axes[0].bar(x, rouge2, width=width, label="ROUGE-2")
axes[0].bar([i + width for i in x], rougeL, width=width, label="ROUGE-L")
axes[0].set_title("ROUGE Score Comparison")
axes[0].set_xticks(x)
axes[0].set_xticklabels([m.upper() for m in models])
axes[0].set_ylabel("ROUGE F1 Score")
axes[0].legend()
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Precision/Recall/F1 subplot
axes[1].bar([i - width for i in x], precision, width=width, label="Precision")
axes[1].bar(x, recall, width=width, label="Recall")
axes[1].bar([i + width for i in x], f1, width=width, label="F1-Score")
axes[1].set_title("Precision, Recall, and F1 Score")
axes[1].set_xticks(x)
axes[1].set_xticklabels([m.upper() for m in models])
axes[1].set_ylabel("Score")
axes[1].legend()
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

plt.suptitle("Summarization Model Evaluation Metrics", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the figure
combined_path = os.path.join("evaluation", "combined_evaluation_plot.png")
plt.savefig(combined_path)
plt.show()

print(f"✅ Combined evaluation plot saved to: {combined_path}")
