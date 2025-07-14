# Airline Sentiment Analysis

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange.svg)](https://jupyter.org/)

## Overview
This project performs sentiment analysis on airline customer tweets to classify them into positive, negative, or neutral sentiments. The analysis includes comprehensive data preprocessing, class balancing techniques, and comparison of multiple machine learning models.

## Dataset
- **Source**: Twitter Airline Sentiment dataset from Kaggle
- **Content**: Customer tweets about airline experiences
- **Classes**: Positive, Negative, Neutral sentiment

## Key Features

### 1. Data Preprocessing
- Text normalization (lowercase conversion)
- URL removal
- User mentions (@users) removal
- Hashtag and punctuation removal
- Number removal
- Stopwords filtering
- TF-IDF vectorization with n-grams

### 2. Class Distribution Analysis
- Detailed analysis of class imbalance
- Visualization of sentiment distribution
- Imbalance ratio calculation

### 3. Class Balancing Techniques
- **SMOTE** (Synthetic Minority Oversampling Technique)
- **Random Under Sampling**
- **SMOTEENN** (Combined over and under sampling)

### 4. Model Comparison
- **Naive Bayes**
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest**

### 5. Advanced Analysis
- Confusion matrix visualization
- Classification reports
- Word cloud generation by sentiment
- Model performance comparison with balanced data

## Results
The project systematically compares different balancing techniques and models to achieve optimal sentiment classification performance. The final model selection is based on accuracy, F1-score, precision, and recall metrics.

## Technologies Used
- **Python 3.x**
- **Libraries**:
  - pandas, numpy
  - scikit-learn
  - imbalanced-learn
  - matplotlib, seaborn
  - wordcloud
  - nltk

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
wordcloud
nltk
kaggle
```

## Usage
1. Install required packages: `pip install -r requirements.txt`
2. Set up Kaggle API credentials
3. Run the Jupyter notebook `sentiment_analysis_airlines.ipynb`

## Project Structure
```
sentiment_analysis_airlines/
├── sentiment_analysis_airlines.ipynb  # Main analysis notebook
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
├── Tweets.csv                         # Dataset (downloaded from Kaggle)
└── database.sqlite                    # SQLite database
```

## Key Insights
- Dataset shows significant class imbalance favoring negative sentiments
- Class balancing techniques significantly improve model performance
- TF-IDF with n-grams provides better feature representation
- Model selection depends on the specific balancing technique used

## Author
Created as part of a machine learning sentiment analysis project.

## License
This project is for educational and research purposes.
