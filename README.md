# 🧠 Product Sentiment Analysis using Amazon Reviews

##  Overview
This project performs **sentiment analysis on product reviews** using NLP and Machine Learning techniques.  
It classifies reviews into:
- Positive  
- Negative  
- Neutral  

The system combines:
- **BERT embeddings** for feature extraction  
- **Machine Learning models** (SVM, Random Forest, XGBoost)  
- **Ensemble learning** for improved performance  

---

## Dataset
- **Source:** Amazon Reviews Dataset (McAuley Lab, 2023)
- **Category:** All_Beauty
- **Size:** ~701K reviews  

### Features:
- `rating`
- `title`
- `text`
- `asin`
- `user_id`
- `timestamp`
- `verified_purchase`

### Class Distribution:
- Positive: ~70%
- Negative: ~20%
- Neutral: ~8%

---

## ⚙️ System Architecture

### Pipeline Diagram
             ┌──────────────────────────────┐
             │        DATA SOURCE           │
             │ Amazon Reviews Dataset       │
             └──────────────┬───────────────┘
                            │
                            ▼
             ┌──────────────────────────────┐
             │      DATA PREPROCESSING      │
             │ - Combine title + text       │
             │ - Remove null/short reviews  │
             │ - Text cleaning              │
             └──────────────┬───────────────┘
                            │
                            ▼
             ┌──────────────────────────────┐
             │    FEATURE EXTRACTION        │
             │                              │
             │  ┌──────────────┐            │
             │  │   TF-IDF     │            │
             │  └──────────────┘            │
             │  ┌──────────────┐            │
             │  │    BERT      │            │
             │  └──────────────┘            │
             └──────────────┬───────────────┘
                            │
                            ▼
    ┌────────────────────────────────────────────┐
    │            MODEL TRAINING                  │
    │                                            │
    │   ┌──────────┐  ┌──────────┐  ┌──────────┐ │
    │   │   SVM    │  │ Random   │  │ XGBoost  │ │
    │   │          │  │ Forest   │  │          │ │
    │   └──────────┘  └──────────┘  └──────────┘ │
    └──────────────┬─────────────────────────────┘
                   │
                   ▼
    ┌────────────────────────────────────────────┐
    │          ENSEMBLE MODEL                    │
    │   (Combines predictions of all models)     │
    └──────────────┬─────────────────────────────┘
                   │
                   ▼
    ┌────────────────────────────────────────────┐
    │           PREDICTION OUTPUT                │
    │   Positive / Negative / Neutral            │
    └────────────────────────────────────────────┘

    
---

## 🧹 Data Preprocessing
- Combined `title + text`
- Removed empty and short reviews
- Handled missing values
- Cleaned punctuation and noise

---

## 🔍 Feature Extraction

### TF-IDF
- Converts text into numerical vectors
- Works well with traditional ML models

### BERT
- Contextual embeddings
- Handles multilingual & informal text
- Captures semantic meaning

---

## 🤖 Models Used

### 1. Support Vector Machine (SVM)
- Handles high-dimensional data
- Used with class balancing

### 2. Random Forest
- Ensemble of decision trees
- Good interpretability

### 3. XGBoost
- Gradient boosting algorithm
- Handles imbalance effectively

### 4. Ensemble Model
- Combines all models
- Improves robustness

---

## 📈 Results

| Model          | Accuracy |
|----------------|----------|
| SVM            | 86%      |
| Random Forest  | 82%      |
| XGBoost        | 81%      |
| Ensemble       | 85%      |

---

## Challenges & Solutions

### Class Imbalance
- Used class weights (SVM)
- Used scale_pos_weight (XGBoost)

### Noisy Data
- Removed empty/short reviews
- Combined text fields

### Large Dataset
- Batch processing for BERT
- Cached embeddings

### Multilingual Text
- Handled using BERT

---

## 🚀 How to Run

```bash
# Clone repo
git clone https://github.com/your-username/product-sentiment-analysis.git

cd product-sentiment-analysis

# Install dependencies
pip install -r requirements.txt

# Train model
python train.py

# Run predictions
python predict.py

├── data/
├── notebooks/
├── models/
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train.py
│   └── predict.py
├── requirements.txt
└── README.md
