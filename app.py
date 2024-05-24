from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the trained model and tokenizer
model = load_model('sentiment_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    data = request.json
    text = data.get("text", "")
    
    # Preprocess the input text
    sequences = tokenizer.texts_to_sequences([text])
    maxlen = 100
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    
    # Debugging: Print input sequences and padded sequences
    print("Input text:", text)
    print("Tokenized sequences:", sequences)
    print("Padded sequences:", padded_sequences)
    
    # Predict sentiment
    prediction = model.predict(padded_sequences)[0][0]
    
    # Debugging: Print prediction score
    print("Prediction score:", prediction)
    
    # Experiment with threshold adjustment
    sentiment = "pos" if prediction > 0.6 else "neg"
    
    response = {
        "text": text,
        "sentiment": sentiment
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
