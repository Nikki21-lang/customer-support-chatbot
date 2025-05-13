
# chatbot.py
import random
import json
import nltk # type: ignore
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, request, jsonify

nltk.download('punkt')

# Sample training data
intents = {
    "intents": [
        {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey", "Good morning"], "responses": ["Hello! How can I help you today?", "Hi there!"]},
        {"tag": "goodbye", "patterns": ["Bye", "Goodbye", "See you later"], "responses": ["Goodbye! Have a nice day.", "See you later!"]},
        {"tag": "thanks", "patterns": ["Thanks", "Thank you", "Appreciate it"], "responses": ["You're welcome!", "Glad I could help."]},
        {"tag": "order_status", "patterns": ["Where is my order?", "Track my order", "Order status"], "responses": ["Can I have your order ID?", "Let me check your order details."]},
        {"tag": "refund", "patterns": ["I want a refund", "Return product", "Money back"], "responses": ["Sure, I can help you with that. Please share your order number."]}
    ]
}

# Prepare data
patterns = []
tags = []
responses = {}

for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
    responses[intent['tag']] = intent['responses']

# Tokenize and vectorize
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)
y = tags

# Train a basic classifier
model = MultinomialNB()
model.fit(X, y)

# Chatbot function
def chatbot_response(message):
    input_vect = vectorizer.transform([message])
    predicted_tag = model.predict(input_vect)[0]
    return random.choice(responses[predicted_tag])

# Flask API
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No input provided."}), 400

    reply = chatbot_response(user_input)
    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(debug=True)
