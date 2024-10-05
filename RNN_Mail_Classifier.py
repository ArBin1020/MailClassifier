import os
import re
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Dense, LSTM
nltk.download('stopwords')

MODEL_FILE = 'spam_classifier_rnn_model.h5'
TOKENIZER_FILE = 'tokenizer.pkl'

def preprocess_text(message):
    ps = PorterStemmer()
    message = re.sub('[^a-zA-Z]', ' ', message)
    message = message.lower()
    message = message.split()
    message = [ps.stem(word) for word in message if word not in stopwords.words('english')]
    message = ' '.join(message)
    return message

def train_model():
    messages = pd.read_csv('spam.csv', encoding='ISO-8859-1')
    
    messages = messages[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
    
    corpus = [preprocess_text(message) for message in messages['message']]
    
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(corpus)
    X = tokenizer.texts_to_sequences(corpus)
    
    X = pad_sequences(X, maxlen=100)
    
    y = pd.get_dummies(messages['label'], drop_first=True).values.ravel()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
    model.add(LSTM(units=128, return_sequences=False))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))
    
    model.save(MODEL_FILE)
    with open(TOKENIZER_FILE, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print(f"Model training completed, accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {fscore:.3f}")

def predict_spam(input_message):
    if not os.path.exists(MODEL_FILE) or not os.path.exists(TOKENIZER_FILE):
        print("Model not found, training in progress...")
        train_model()
    
    
    model = load_model(MODEL_FILE)
    with open(TOKENIZER_FILE, 'rb') as f:
        tokenizer = pickle.load(f)
    
    preprocessed_message = preprocess_text(input_message)
    
    input_sequence = tokenizer.texts_to_sequences([preprocessed_message])
    input_sequence = pad_sequences(input_sequence, maxlen=100)
    
    prediction = (model.predict(input_sequence) > 0.5).astype("int32")
    
    if prediction[0] == 1:
        return "This is a spam email"
    else:
        return "This is a normal email"

if __name__ == "__main__":
    while True:
        input_message = input("Please enter a message to determine if it is spam (type 'exit' to leave): ")
        if input_message.lower() == 'exit':
            break
        result = predict_spam(input_message)
        print(result)
