import os
import re
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

nltk.download('stopwords')

MODEL_FILE = 'spam_classifier_model.pkl'
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'

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
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(corpus).toarray()
    
    y = pd.get_dummies(messages['label'], drop_first=True).values.ravel()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print(f"Model training completed, accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {fscore:.3f}")

    from sklearn.metrics import roc_curve, auc

    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def predict_spam(input_message):
    if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
        print("Model not found, training in progress...")
        train_model()
    
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_FILE, 'rb') as f:
        vectorizer = pickle.load(f)
    
    preprocessed_message = preprocess_text(input_message)
    
    input_features = vectorizer.transform([preprocessed_message]).toarray()
    
    prediction = model.predict(input_features)
    
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
