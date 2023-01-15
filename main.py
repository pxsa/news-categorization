import streamlit as st
import numpy as np
import pandas as pd
import codecs
from hazm import WordTokenizer, Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


#
trainDF = pd.read_csv('data/cleanTrain.csv')
testDF  = pd.read_csv('data/cleanTest.csv')

#
def removeUnnecessaryChars(df):
    lst = np.array(['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹', '؛', ':', '،', '!', '؟', '.'])
    contents = []
    
    for i in range(df.shape[0]):
        container = df['content'][i]
        # remove special characters
        for char in lst:
            container = container.replace(char, "")
        # remove half-space
        container = container.replace('\u200c', " ")
        # change 2 spaces with single space
        container = container.replace('  ', " ")
        contents.append(container)

    return contents

#
trainDF.content = removeUnnecessaryChars(trainDF)
testDF.content  = removeUnnecessaryChars(testDF)

#
nmz = Normalizer()
stops = "\n".join(
    sorted(
        list(
            set(
                [
                    nmz.normalize(w) for w in codecs.open('persian-stopwords-master/persian', encoding='utf-8').read().split('\n') if w
                ]
            )
        )
    )
)
stops = stops.split('\n')

# Feature Extraction
vectorizer = TfidfVectorizer(stop_words = stops, tokenizer = WordTokenizer().tokenize)
vectorizer = vectorizer.fit(trainDF.content.values)
train = vectorizer.transform(trainDF.content.values)
test  = vectorizer.transform(testDF.content.values)

# Feature Selection
feature_selector = VarianceThreshold(threshold=1e-5)
feature_selector = feature_selector.fit(train)
x_train = feature_selector.transform(train)
x_test  = feature_selector.transform(test)
y_train = trainDF.category.values
y_test  = testDF.category.values

# Model
def createModel(x, y):
    return SVC().fit(x, y)

# Check
def check(content, svc):
    if svc:
        data = {'content': content}
        df_predict = pd.DataFrame(data, index=[0])

        df_predict.content  = removeUnnecessaryChars(df_predict)
        sampleTest  = vectorizer.transform(df_predict.content.values)
        sample_x_test  = feature_selector.transform(sampleTest)

        category = svc.predict(sample_x_test)
        st.write(category)
    else:
        st.write("please train model first")



st.title("News Classification")
agree = st.checkbox('Train model')
content = st.text_area("news content")
svc = None
if agree:
    svc = createModel(x_train, y_train)
    st.write("model trained successfully")
    
if st.button("ok"):
    check(content, svc)
