import os
import matplotlib.pyplot as plt

# Load the saved ROUGE scores
results_file = os.path.join("evaluation", "rouge_scores.txt")

# Data structure to hold scores
scores = {
    "baseline": {},
    "lsa": {},
    "lexrank": {},
    "kmedoid": {}
}

# Parse the txt file
current_model = None
with open(results_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith("🔹"):
            current_model = line[2:].strip().lower()
        elif "ROUGE-1" in line:
            scores[current_model]["rouge1"] = float(line.split(":")[1])
        elif "ROUGE-2" in line:
            scores[current_model]["rouge2"] = float(line.split(":")[1])
        elif "ROUGE-L" in line:
            scores[current_model]["rougeL"] = float(line.split(":")[1])

# Build bar chart
models = list(scores.keys())
rouge1 = [scores[m]["rouge1"] for m in models]
rouge2 = [scores[m]["rouge2"] for m in models]
rougeL = [scores[m]["rougeL"] for m in models]

x = range(len(models))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar([i - width for i in x], rouge1, width=width, label="ROUGE-1")
plt.bar(x, rouge2, width=width, label="ROUGE-2")
plt.bar([i + width for i in x], rougeL, width=width, label="ROUGE-L")

plt.xlabel("Summarization Models")
plt.ylabel("ROUGE F1 Score")
plt.title("ROUGE Score Comparison")
plt.xticks(x, [m.upper() for m in models])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save plot
plot_path = os.path.join("evaluation", "rouge_plot.png")
plt.tight_layout()
plt.savefig(plot_path)
plt.show()

print(f"✅ Plot saved to: {plot_path}")
