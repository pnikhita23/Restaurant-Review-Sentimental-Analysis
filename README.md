

# Restaurant Review Sentiment Analysis

This project focuses on sentiment analysis of restaurant reviews using various machine learning techniques. The objective is to classify reviews as positive or negative based on their content.

## Overview

- **Dataset:** The dataset consists of 1,000 rows and 2 columns:
  - `Review`: Contains the text of the customer review.
  - `Liked`: Binary target variable, where `1` indicates a positive review and `0` indicates a negative review.
- **Goal:** To build a classification model that accurately predicts the sentiment of restaurant reviews.
- **Best Accuracy Achieved:** 76.6% using Logistic Regression.

## Project Structure

1. **Data Preprocessing**
   - Checked for missing values and found none.
   - Tokenization, stopword removal, and stemming were applied to clean the text data.
   - Extracted features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

2. **Exploratory Data Analysis (EDA)**
   - Performed text-based analysis such as character count, word count, and sentence count for each review.
   - Created word clouds for positive and negative reviews to visualize frequently used words.

3. **Model Training and Evaluation**
   - Split the dataset into training and testing sets.
   - Tried multiple models, including:
     - Logistic Regression (best accuracy: 76.6%)
     - Naive Bayes
     - Random Forest
   - Evaluated model performance using metrics like accuracy, precision, recall, F1-score, and confusion matrix.

4. **Model Deployment**
   - Saved the trained Logistic Regression model and the TF-IDF vectorizer using pickle for future use.
   
## Usage

1. **Preprocess the data:** Run the notebook to clean the text data and vectorize using TF-IDF.
2. **Train the model:** Train different models and evaluate their performance.
3. **Make predictions:** Use the trained Logistic Regression model to predict the sentiment of new reviews.

## Results

- **Best Model:** Logistic Regression
- **Accuracy:** 76.6%
- **Confusion Matrix:** Shows the breakdown of true positives, true negatives, false positives, and false negatives.

## Future Work

- Explore more sophisticated NLP techniques such as word embeddings (e.g., Word2Vec, GloVe).
- Use deep learning models like LSTM or BERT for better accuracy.
- Collect more data to improve the model's generalization.

## Technologies Used

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, NLTK, Matplotlib, Seaborn, WordCloud

