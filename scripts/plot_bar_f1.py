import os
import matplotlib.pyplot as plt

# Load evaluation results
results_file = os.path.join("evaluation", "rouge_scores.txt")

# Structure to store scores
scores = {
    "baseline": {},
    "lsa": {},
    "lexrank": {},
    "kmedoid": {}
}

# Parse evaluation file
current_model = None
with open(results_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith("🔹"):
            current_model = line[2:].strip().lower()
        elif "Precision" in line and current_model:
            try:
                scores[current_model]["precision"] = float(line.split(":")[1].strip())
            except ValueError:
                scores[current_model]["precision"] = 0.0
        elif "Recall" in line and current_model:
            try:
                scores[current_model]["recall"] = float(line.split(":")[1].strip())
            except ValueError:
                scores[current_model]["recall"] = 0.0
        elif "F1-Score" in line and current_model:
            try:
                scores[current_model]["f1"] = float(line.split(":")[1].strip())
            except ValueError:
                scores[current_model]["f1"] = 0.0


# Prepare data for plot
models = list(scores.keys())
precision = [scores[m]["precision"] for m in models]
recall = [scores[m]["recall"] for m in models]
f1 = [scores[m]["f1"] for m in models]

x = range(len(models))
width = 0.25

# Plotting
plt.figure(figsize=(10, 6))
plt.bar([i - width for i in x], precision, width=width, label="Precision")
plt.bar(x, recall, width=width, label="Recall")
plt.bar([i + width for i in x], f1, width=width, label="F1-Score")

plt.xlabel("Summarization Models")
plt.ylabel("Score")
plt.title("Precision, Recall, and F1-Score Comparison")
plt.xticks(x, [m.upper() for m in models])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save plot
plot_path = os.path.join("evaluation", "f1_plot.png")
plt.tight_layout()
plt.savefig(plot_path)
plt.show()

print(f"✅ Precision/Recall/F1 plot saved to: {plot_path}")
