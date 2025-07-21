import os

# Number of sentences to include in the summary
k = 5

# Set input/output paths
input_dir = os.path.join("data", "docs")
output_dir = os.path.join("outputs", "baseline")
os.makedirs(output_dir, exist_ok=True)

# Loop through each text document
for index, filename in enumerate(sorted(os.listdir(input_dir))):
    if filename.startswith("."):
        continue
    print(f"Processing file: {filename}")  # 👈 Add this line



    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f"summary_{index}.txt")

    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        summary_lines = lines[:k]  # get first k lines

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for line in summary_lines:
            outfile.write(line.strip() + '\n')
