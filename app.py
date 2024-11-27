from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from transformers import pipeline
import numpy as np

# Flask app
app = Flask(__name__)

# Load pre-trained models
# Load the trained SVM model, TF-IDF vectorizer, and Label Encoder
svm_model = pickle.load(open("svm_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Initialize zero-shot classification pipeline
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Helper function to clean tweets
def clean_tweet(tweet):
    return tweet.strip().replace("@", ",")  # Example cleaning: remove spaces and replace "@"

# Route for the home page
@app.route('/')
def home():
    return render_template("index.html")  # The form will be in `index.html`

# Route to predict the cyberbullying type
@app.route('/predict', methods=['POST'])
def predict():
    # Get tweet input from the form
    tweet_input = request.form.get("tweet")

    if not tweet_input:
        return jsonify({"error": "Please provide a tweet."}), 400

    # Clean and transform the input
    tweet_input_cleaned = clean_tweet(tweet_input)
    tweet_input_tfidf = tfidf.transform([tweet_input_cleaned])

    # Predict using the SVM model
    predicted_label = svm_model.predict(tweet_input_tfidf)
    predicted_category = label_encoder.inverse_transform(predicted_label)[0]

    # If "not_cyberbullying", use zero-shot classifier for additional check
    if predicted_category == "not_cyberbullying":
        labels = ["bullying", "not bullying"]
        result = zero_shot_classifier(tweet_input, candidate_labels=labels)

        # Check the highest label
        highest_label = result['labels'][0]
        if highest_label == "bullying":
            predicted_category = "other_cyberbullying"

    # Return the prediction on the same page
    return render_template("index.html", tweet=tweet_input, predicted_category=predicted_category)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
