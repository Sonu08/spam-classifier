import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier


rfc = RandomForestClassifier(n_estimators=50, random_state=2)

def transform_text(text):
    text = text.lower() # lower case
    
    text = WordTokenizer(text) # tokenization
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i) # removing special characters
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i) # removing stop words and punctuations
            
    text = y[:]
    y.clear()
    
    ps = PorterStemmer()
    
    
    for i in text:
        y.append(ps.stem(i)) # stemming
    
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

# 1. Preprocess
transformed_sms = transform_text(input_sms)

# 2. Vectorization
vector_input = tfidf.transform([transformed_sms])

# 3. Predict
result = model.predict(vector_input)[0]

# 4. Display
if result == 1:
    st.haeder("Spam")
else:
    st.header("Not Spam")

