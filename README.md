# 🧠 Comparative Text Summarization System

A powerful **unsupervised text summarization toolkit** that implements and compares multiple classic algorithms. This project allows users to generate summaries using various extractive methods and evaluate them automatically using standard metrics.

---

## 🚀 Features

- 📄 **Multiple Summarization Algorithms**
  - TextRank
  - LexRank
  - LSA (Latent Semantic Analysis)
  - SumBasic
- 📊 **Automated Evaluation**
  - ROUGE Score
  - Cosine Similarity
  - Compression Ratio
- ⚙️ **Modular & Extensible**
  - Easy to add new algorithms or evaluation metrics
- 📈 **Optional Visualizations**
  - Summary quality plots
  - Length and compression statistics

---

## 🛠 Tech Stack

- **Python 3.8+**
- `nltk`, `scikit-learn`, `networkx`, `sumy`, `rouge-score`, `numpy`, `matplotlib`

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/Text_Summarization.git
cd Text_Summarization
pip install -r requirements.txt
```
## ⚡ Usage
```bash
python main.py --input_file data/sample_text.txt --summary_length 5
```

Arguments:
--input_file: Path to your text file

--summary_length: Number of sentences in each summary (default: 5)

## 🧩 Project Structure
```bash
Text_Summarization/
├── data/
│   ├── docs/                       # Input documents
│   │   └── news_1.txt
│   ├── references/                # Reference summaries for evaluation
│   │   └── news_1_reference.txt
│   └── stopwords.txt              # Stopword list
│
├── evaluation/                    # Evaluation scores and visualizations
│   ├── combined_evaluation_plot.png
│   ├── f1_plot.png
│   ├── plot_bar.png
│   ├── rouge_plot.png
│   └── rouge_scores.txt
│
├── models/                        # Supporting data/models
│   └── idfs_model.txt
│
├── outputs/                       # Generated summaries
│   ├── baseline/
│   ├── kmedoid/
│   ├── lexrank/
│   └── lsa/
│       ├── summary_0.txt
│       └── summary_1.txt
│
├── scripts/                       # Core algorithm scripts
│   ├── baseline_summarizer.py
│   ├── Clustering.py
│   ├── evaluate_rouge.py
│   ├── idf_model.py
│   ├── Kmedoid_summarize.py
│   ├── LexRank.py
│   ├── LSA_summary.py
│   ├── plot_bar_combined.py
│   ├── plot_bar_f1.py
│   └── plot_bar.py
│
├── README.md
└── requirements.txt

```

## 🎯 Future Improvements
- Add abstractive summarization models (T5, BART)

- Support multilingual summarization

- Add web interface for interactive use

- Additional metrics: METEOR, BLEU, BERTScore


