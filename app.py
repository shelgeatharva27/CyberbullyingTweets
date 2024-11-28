from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from transformers import pipeline
import random

# Flask app
app = Flask(__name__)

# Load pre-trained models
svm_model = pickle.load(open("svm_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Initialize zero-shot classification pipeline
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Helper function to clean tweets
def clean_tweet(tweet):
    return tweet.strip().replace("@", ",")  # Example cleaning: remove spaces and replace "@"

# Route for the lander page
@app.route('/')
def lander():
    return render_template("lander.html")

# Route for the chatbot page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template("index.html")

    # Get tweet input from the AJAX request
    data = request.get_json()
    tweet_input = data.get("tweet", "")

    if not tweet_input:
        return jsonify({"error": "Please provide a tweet."}), 400

    # Clean and transform the input
    tweet_input_cleaned = clean_tweet(tweet_input)
    tweet_input_tfidf = tfidf.transform([tweet_input_cleaned])

    # Predict using the SVM model
    predicted_label = svm_model.predict(tweet_input_tfidf)
    predicted_category = label_encoder.inverse_transform(predicted_label)[0]

    # List of possible responses for each category
    bullying_responses = [
        "This tweet is clearly cyberbullying. It contains harmful language targeting others.",
        "Unfortunately, this tweet falls under the category of cyberbullying. It can cause emotional harm.",
        "This tweet is an example of cyberbullying, where someone is intentionally being harmful.",
        "This tweet appears to be bullying. It targets someone with harmful intent.",
        "Cyberbullying is evident in this tweet. It negatively affects the target's mental health.",
        "This is an example of bullying behavior, where the intent is to hurt or upset someone.",
        "The language used in this tweet is hurtful and intended to cause emotional distress.",
        "This tweet fits the definition of cyberbullying. It is aggressive and hurtful in nature.",
        "The tone of this tweet is aggressive, and it seems meant to demean someone publicly.",
        "This tweet is a form of bullying and can have serious consequences for the person targeted."
    ]
    
    not_bullying_responses = [
        "This tweet doesn't appear to be cyberbullying. It seems harmless.",
        "The tweet is not classified as cyberbullying, but context may change its nature.",
        "This tweet is not bullying, but it might require further analysis.",
        "This tweet does not contain harmful intent and does not seem to target anyone.",
        "There is no sign of cyberbullying in this tweet. It appears to be neutral in tone.",
        "This tweet seems like an ordinary statement without any intention to harm anyone.",
        "No bullying detected in this tweet. It does not show aggressive or harmful language.",
        "The language in this tweet does not indicate any form of bullying or harassment.",
        "This tweet seems innocent. There are no signs of harm or malicious intent.",
        "This tweet appears to be non-aggressive and does not seem intended to cause harm."
    ]
    
    other_cyberbullying_responses = [
        "This tweet doesn't seem to be bullying at first glance, but a deeper look suggests it could be harmful.",
        "Though initially it seems harmless, this tweet can be categorized under 'other' forms of cyberbullying.",
        "This is an example of indirect or subtle cyberbullying. The tone may be more passive-aggressive.",
        "While the tweet does not explicitly target someone, it may be considered harmful or demeaning in a subtle way.",
        "The tweet may appear benign, but it can still be a form of cyberbullying due to its underlying tone.",
        "This tweet is an example of passive-aggressive behavior, which can be equally harmful as direct bullying.",
        "It is not immediately obvious, but the content of this tweet may have a negative impact on someone's mental well-being.",
        "While not direct bullying, the tweet can be seen as undermining or belittling someone in a subtle way.",
        "The nature of this tweet is more covert. It's an example of emotional manipulation or subtle harassment.",
        "Though not overtly bullying, this tweet could be harmful as it uses language that diminishes someone's worth or value."
    ]
    
    explanation = ""

    # Select response based on category
    if predicted_category == "bullying":
        explanation = random.choice(bullying_responses)
    elif predicted_category == "not_cyberbullying":
        # Use zero-shot classification to confirm
        labels = ["bullying", "not bullying"]
        result = zero_shot_classifier(tweet_input, candidate_labels=labels)

        highest_label = result['labels'][0]
        if highest_label == "bullying":
            predicted_category = "other_cyberbullying"
            explanation = random.choice(other_cyberbullying_responses)
        else:
            explanation = random.choice(not_bullying_responses)
    else:
        explanation = random.choice(other_cyberbullying_responses)

    # Return the prediction and explanation as a JSON response
    return jsonify({
        "predicted_category": predicted_category,
        "explanation": explanation
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
