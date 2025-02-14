import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
import re
import numpy as np
import random as rn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix

try:
    nltk.download('stopwords')
except Exception as e:
    print("Error downloading stopwords:", e)

def get_stopwords():

    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words("english"))
        
    except Exception as e:
        print("NLTK stopwords not available due to:", e)
        print("Using fallback stopwords list.")
        stop_words = {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
            "your", "yours", "yourself", "yourselves", "he", "him", "his",
            "himself", "she", "her", "hers", "herself", "it", "its", "itself",
            "they", "them", "their", "theirs", "themselves", "what", "which",
            "who", "whom", "this", "that", "these", "those", "am", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had", "having",
            "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
            "or", "because", "as", "until", "while", "of", "at", "by", "for",
            "with", "about", "against", "between", "into", "through", "during",
            "before", "after", "above", "below", "to", "from", "up", "down", "in",
            "out", "on", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "any",
            "both", "each", "few", "more", "most", "other", "some", "such", "no",
            "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
            "t", "can", "will", "just", "don", "should", "now"
        }
    return stop_words

stop_words = get_stopwords()
ps = PorterStemmer()

def remove_html_tags_bs(text):
    return BeautifulSoup(text, "html.parser").get_text()

def clean_text(review):
    review = review.lower()
    review = re.sub(r"[^a-zA-Z0-9]", " ", review)
    review = review.split()
    review = [word for word in review if word not in stop_words]
    review = " ".join([ps.stem(word) for word in review])
    return review

def plot_class_distribution(data):
    class_dist = data['sentiment'].value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_dist.index, y=class_dist.values, palette="Blues")
    plt.title("Class Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

def plot_wordclouds(data):

    positive_reviews = " ".join(data[data['sentiment'] == 'positive']["review"])
    negative_reviews = " ".join(data[data['sentiment'] == 'negative']["review"])

    plt.figure(figsize=(10, 10))
    plt.title("Word Cloud of Positive Reviews")
    wordcloud_pos = WordCloud(background_color='white', max_words=200).generate(positive_reviews)
    plt.imshow(wordcloud_pos, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.title("Word Cloud of Negative Reviews")
    wordcloud_neg = WordCloud(background_color='black', max_words=200).generate(negative_reviews)
    plt.imshow(wordcloud_neg, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

def plot_training_loss(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

def plot_confusion_matrix_func(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def load_and_preprocess_data(
    csv_path,
    seed_value=1337,
    max_words=10000,
    max_len=200,
    test_size=0.2
):

    np.random.seed(seed_value)
    rn.seed(seed_value)
    import tensorflow as tf
    tf.random.set_seed(seed_value)

    data = pd.read_csv(csv_path)

    plot_class_distribution(data)

    data['review'] = data['review'].apply(remove_html_tags_bs)

    X = data['review'].copy()
    y = data['sentiment'].copy()

    X_cleaned = X.apply(clean_text)
    y_binary = y.map({'positive': 1, 'negative': 0})

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_cleaned, y_binary, test_size=test_size, random_state=seed_value
    )

    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)

    train_sequences = tokenizer.texts_to_sequences(X_train)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    train_padded = pad_sequences(train_sequences, maxlen=max_len,
                                 padding='post', truncating='post')
    test_padded = pad_sequences(test_sequences, maxlen=max_len,
                                padding='post', truncating='post')

    return data, train_padded, test_padded, y_train, y_test, tokenizer
