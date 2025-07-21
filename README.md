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
git clone https://github.com/yourusername/comparative-text-summarizer.git
cd comparative-text-summarizer
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
comparative-text-summarizer/
├── algorithms/
│   ├── textrank.py
│   ├── lexrank.py
│   ├── lsa.py
│   └── sumbasic.py
├── evaluation/
│   ├── rouge.py
│   ├── similarity.py
│   └── compression.py
├── data/
│   └── sample_text.txt
├── outputs/
├── main.py
├── requirements.txt
└── README.md
```

## 🎯 Future Improvements
- Add abstractive summarization models (T5, BART)

- Support multilingual summarization

- Add web interface for interactive use

- Additional metrics: METEOR, BLEU, BERTScore


