# Malay-Indonesian False Friend Disambiguator

An NLP project designed to distinguish between identical words with different meanings in Indonesian and Malay using Statistical and Transformer-based models.

## 🚀 Project Overview
This project implements a system to disambiguate "False Friends"—words like *Kereta*, *Bisa*, and *Budak*—by identifying the language of the surrounding context.

### Features
- **3-Tier Model Implementation**: Comparison between TF-IDF, Hybrid Lexical models, and Multilingual BERT.
- **Interactive Demo**: A colorful, real-time terminal interface for testing custom sentences.
- **Detailed Methodology**: A comprehensive report covering the design, pipeline, and experimental results.

## 📂 Project Structure
- `false_friend_model.py`: Baseline implementation (TF-IDF + Logistic Regression/Naive Bayes).
- `false_friend_bert.py`: Advanced implementation using Multilingual BERT embeddings.
- `false_friend_demo.py`: Interactive terminal interface for live testing.
- `methodology_report.md`: The formal methodology write-up (PDF/Markdown).
- `preprocessed_false_friends_dataset.csv`: The underlying dataset used for training.

## 🛠️ Installation
1. Ensure you have Python 3.10+ installed.
2. Install the required dependencies:
```bash
pip install pandas scikit-learn transformers torch colorama tqdm joblib
```

## 📖 Usage
### 1. Run the Baseline/Hybrid Model
```bash
python false_friend_model.py
```
### 2. Run the Advanced BERT Feature Extractor (Baseline)
```bash
python false_friend_bert.py
```

### 3. Fine-Tune the IndoBERT Model End-to-End
To fine-tune the high-accuracy `indobenchmark/indobert-base-p1` model:
```bash
python fine_tune_bert.py
```
*Note: This will train the model weights end-to-end on CPU (takes ~4 minutes) and save it to the `fine_tuned_bert/` folder.*

### 4. Start the Interactive Demo
```bash
python false_friend_demo.py
```

## 📊 Experimental Results
- **Baseline Accuracy (Logistic Regression)**: 72.00%
- **Hybrid Features Accuracy (Naive Bayes)**: 74.00%
- **mBERT Contextual Feature Extraction**: 76.50%
- **Fine-Tuned IndoBERT (End-to-End)**: **82.00%** 🏆

## 📝 License
This project was developed as part of Assignment 3b: Model Implementation (Methodology).
