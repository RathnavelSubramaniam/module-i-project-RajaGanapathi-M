# AI-Driven Sentiment Analysis for Stock Market News

An end-to-end NLP project that classifies the sentiment of stock-related news articles (Positive / Neutral / Negative) using multiple text embedding techniques and machine learning / deep learning models, then compares them to identify the best-performing approach.

## Problem Statement

### Business Context
Stock prices are heavily influenced by company performance, innovations, collaborations, and market sentiment. News and media coverage can rapidly shift investor perception and, in turn, stock prices. With the sheer volume of financial news published daily, investors and analysts struggle to keep up and accurately gauge its market impact.

### Problem Definition
An investment startup has collected historical daily news for a NASDAQ-listed company, along with its daily stock price (Open, High, Low, Close) and trading volume. The goal of this project is to build an AI-driven sentiment analysis system that automatically classifies news sentiment, helping analysts translate news flow into actionable investment insight.

## Dataset

| Column | Description |
|---|---|
| `Date` | Date the news was released |
| `News` | Content of the news article |
| `Open` | Stock price ($) at market open |
| `High` | Highest stock price ($) during the day |
| `Low` | Lowest stock price ($) during the day |
| `Close` | Adjusted stock price ($) at market close |
| `Volume` | Number of shares traded during the day |
| `Label` | Sentiment polarity — `1`: Positive, `0`: Neutral, `-1`: Negative |

## Project Workflow

1. **Data Overview & Cleaning** — inspected shape, dtypes, missing values, and duplicates; converted `Date` to datetime.
2. **Exploratory Data Analysis**
   - Univariate: sentiment label distribution, price/volume distributions, news-length distribution, month-wise trends.
   - Bivariate: correlation heatmap, sentiment vs. price, price trends over time, news length across sentiment classes.
3. **Text Embeddings** — three embedding strategies were generated from the news text:
   - **Word2Vec** (trained from scratch, CBOW, 100-dim vectors, document vectors via averaging)
   - **Sentence Transformer — `BAAI/bge-base-en-v1.5`** (768-dim)
   - **Sentence Transformer — `all-MiniLM-L6-v2`** (384-dim)
4. **Modeling** — for each embedding, two classifiers were trained on an 80:20 train/test split (`random_state=42`):
   - **Random Forest Classifier**
   - **Feed-forward Neural Network** (Keras `Sequential`: Dense(128) → Dropout → Dense(64) → Dropout → Dense(3, softmax))
5. **Evaluation** — accuracy, precision, recall, and F1-score (weighted) plus confusion matrices for every model, consolidated into comparison tables and a grouped bar chart.
6. **Prediction** — the best model is used to generate sentiment predictions on sample/new news text.

## Models Compared

| Embedding | Random Forest | Neural Network |
|---|---|---|
| Word2Vec | ✅ | ✅ |
| Sentence Transformer (BAAI/bge-base-en-v1.5) | ✅ | ✅ |
| Sentence Transformer (all-MiniLM-L6-v2) | ✅ | ✅ |

## Results

The **Sentence Transformer (all-MiniLM-L6-v2) + Neural Network** combination was the best performer on the held-out test set:

- **Accuracy:** ~0.58
- **Recall:** ~0.58
- **Precision:** ~0.58
- **F1-score:** ~0.57

Word2Vec-based models and the BAAI Sentence Transformer neural network underperformed (metrics often below 0.45), suggesting that these embeddings captured less discriminative signal for this task/dataset than the MiniLM sentence embeddings.

## Recommendations & Future Work

- Perform more extensive hyperparameter tuning, especially for the neural network models.
- Explore advanced architectures suited to sequential/text data (e.g., RNNs, CNNs, or fine-tuning a transformer encoder end-to-end).
- Try ensemble methods that combine predictions across embeddings/models.
- Expand the training dataset, since the current sample size is relatively small (~349 articles).

## Tech Stack

- **Language:** Python
- **Data Handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Embeddings:** gensim (Word2Vec), sentence-transformers (BAAI/bge-base-en-v1.5, all-MiniLM-L6-v2)
- **Modeling:** scikit-learn (Random Forest), TensorFlow / Keras (Neural Network)
- **Environment:** Google Colab / Jupyter Notebook

