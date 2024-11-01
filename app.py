from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_frozen import Freezer  # Import Frozen-Flask

app = Flask(__name__)

# Load the trained model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the TfidfVectorizer
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)

# Configure Freezer for generating static files
freezer = Freezer(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']  # Get the review from the form

    # Check if the review is not empty
    if review.strip():  # Only process if review is not empty
        # Vectorize the review using the loaded TfidfVectorizer
        review_vectorized = tfidf.transform([review])  

        # Make the prediction
        prediction = model.predict(review_vectorized)

        # Map numerical predictions to sentiment labels
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
    else:
        sentiment = "No Review Given."  # Handle empty input

    return render_template('result.html', sentiment=sentiment)

# Run the freezer to generate static files
if __name__ == '__main__':
    app.config['FREEZER_RELATIVE_URLS'] = True
    freezer.freeze()  # Generate static files