import nltk
from nltk.corpus import movie_reviews
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Download the necessary NLTK data
nltk.download('movie_reviews')

# Load movie reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

print(documents)

# Prepare the data
texts = [" ".join(words) for words, category in documents]
labels = [category for words, category in documents]

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
maxlen = 100
data = pad_sequences(sequences, maxlen=maxlen)

# Split the data
train_X, test_X, train_y, test_y = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=maxlen),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_X, train_y, epochs=5, batch_size=64, validation_split=0.2)

# Save the model and tokenizer
model.save('sentiment_model.h5')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Evaluate the model
predictions = model.predict(test_X)
predictions = (predictions > 0.5).astype(int).reshape(-1)
accuracy = accuracy_score(test_y, predictions)
print(f'Model Accuracy: {accuracy}')

