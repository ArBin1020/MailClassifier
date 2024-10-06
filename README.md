# SMS Spam Classifier
### Introduction
This project is a simple implementation of an SMS spam classifier using machine learning techniques. By leveraging Python and various libraries, we preprocess text data, build a classification model, and evaluate its performance. The goal is to demonstrate the process of creating a functional machine learning model capable of distinguishing between legitimate (ham) and spam messages.

### Purpose
The purpose of this project is to provide a hands-on application of key machine learning concepts, such as data preprocessing, text vectorization, model training, and evaluation. We aim to classify SMS messages as either spam or ham, offering a practical solution for spam detection in text-based communications.

### Dataset
The dataset used in this project is the SMS Spam Collection Dataset, which consists of 5,574 SMS messages, each labeled as either "ham" (legitimate) or "spam". The messages are collected from several sources:

Grumbletext: A UK-based forum where users report SMS spam.
NUS SMS Corpus: A collection of legitimate SMS messages primarily from students in Singapore.
Caroline Tag’s PhD Thesis: A set of SMS ham messages included for research purposes.
SMS Spam Corpus v.0.1 Big: Another set of ham and spam messages publicly available for spam research.

### Installation & Usage Instructions
##### Prerequisites
1. Python 3.x installed
2. Install the necessary libraries:
```bash
pip install pandas nltk scikit-learn matplotlib tensorflow
```
##### Steps to Run the Classifier
1. Download the Dataset
Download the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
2. Clone the Repository
```bash
git clone https://github.com/ArBin1020/MailClassifier.git
cd MailClassifier
```
3. Run the Code
To preprocess the data, train the model, and evaluate the classifier, run the following command:
```bash
python project.py
```
4. Evaluate Results
After running the classifier, the model's performance (accuracy, precision, recall, etc.) will be displayed, along with a visualization of the ROC curve.

