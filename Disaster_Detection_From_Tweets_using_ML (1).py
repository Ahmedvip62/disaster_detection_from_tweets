import pandas as pd
import numpy as np
import nltk
if __name__ == "__main__":
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

import re
import string
from string import punctuation
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
import streamlit as st
from wordcloud import WordCloud


# تحميل البيانات
df = pd.read_csv('disaster_tweets.csv')

# عرض رأس البيانات
st.write(df.head())

# معالجة البيانات
df['length'] = df['text'].apply(len)

# إنشاء عمود جديد للنصوص المعالجة
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)

df['text'] = df['text'].apply(remove_stopwords)

# تنظيف النصوص
def cleanTweet(txt):
    txt = txt.lower()
    words = nltk.word_tokenize(txt)
    words = ' '.join([nltk.WordNetLemmatizer().lemmatize(word) for word in words if word not in stopwords.words('english')])
    txt = re.sub('[^a-z]',' ',words)
    return txt

df['cleaned_tweets'] = df['text'].apply(cleanTweet)

# إنشاء المتغيرات المستهدفة
y = df.target
X = df.cleaned_tweets

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=0)

# TF-IDF Vectorizer - Bi-Gram
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, ngram_range=(1,2))
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# بناء نموذج Multinomial Naive Bayes
mnb_tf = MultinomialNB()
mnb_tf.fit(tfidf_train, y_train)

# التنبؤات
pred_mnb = mnb_tf.predict(tfidf_test)

# النتائج
accuracy = accuracy_score(y_test, pred_mnb)
precision = precision_score(y_test, pred_mnb)
recall = recall_score(y_test, pred_mnb)
f1 = f1_score(y_test, pred_mnb)

# عرض النتائج
st.write(f"Accuracy: {accuracy}")
st.write(f"Precision: {precision}")
st.write(f"Recall: {recall}")
st.write(f"F1 Score: {f1}")

# عرض WordCloud
st.subheader("WordCloud for Disaster Tweets")
wc = WordCloud(max_words=1000, width=1600, height=800).generate(" ".join(df[df.target == 1].text))
st.image(wc.to_array(), use_column_width=True)

# عرض WordCloud ل Tweets العادية
st.subheader("WordCloud for Normal Tweets")
wc = WordCloud(max_words=1000, width=1600, height=800).generate(" ".join(df[df.target == 0].text))
st.image(wc.to_array(), use_column_width=True)

# التنبؤ برسائل عينة
sentences = [
    "Just happened a terrible car crash",
    "Heard about #earthquake is different cities, stay safe everyone.",
    "No I don't like cold!",
    "@RosieGray Now in all sincerety do you think the UN would move to Israel if there was a fraction of a chance of being annihilated?"
]
tfidf_trigram = tfidf_vectorizer.transform(sentences)

predictions = mnb_tf.predict(tfidf_trigram)

for text, label in zip(sentences, predictions):
    target = "Disaster Tweet" if label == 1 else "Normal Tweet"
    st.write(f"Text: {text}")
    st.write(f"Class: {target}")
    st.write("---")






