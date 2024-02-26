# # import streamlit as st
# # import pickle
# # import nltk
# # nltk.download('all')
# # import string
# # from nltk.corpus import stopwords
# # from nltk.stem.porter import PorterStemmer
# # import string

# # ps = PorterStemmer()

# # def transform_text(text):
# #   text = text.lower()
# #   text = nltk.word_tokenize(text)

# #   y =[]
# #   for i in text:
# #     if i.isalnum():
# #       y.append(i)

# #   text = y[:]
# #   y.clear()
# #   for i in text:
# #     if i not in stopwords.words('english') and i not in string.punctuation:
# #       y.append(i)

# #   text = y[:]
# #   y.clear()
# #   for i in text:
# #     y.append(ps.stem(i))

# #   return " ".join(y)

# # tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
# # model = pickle.load(open('model.pkl', 'rb'))

# # st.title("Email/SMS Spam Classifier")

# # input_sms = st.text_area("Enter the message")

# # if st.button('Perdict'):
# #   #1. preprocess
# #   transformed_sms = transform_text(input_sms)
# #   #2. vectorize
# #   vector_input = tfidf.tranform([transformed_sms])
# #   #3. predict 
# #   result = model.predict(vector_input)[0]
# #   #4. Display
# #   if result ==1 :
# #     st.header("Spam")
# #   else:
# #     st.header("Not Spam")

# import streamlit as st
# import pickle
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer

# nltk.download('punkt')
# nltk.download('stopwords')

# ps = PorterStemmer()

# def preprocess_text(text):
#     text = text.lower()
#     words = nltk.word_tokenize(text)
    
#     # Remove non-alphanumeric characters and stopwords
#     words = [ps.stem(word) for word in words if word.isalnum() and word not in stopwords.words('english')]
    
#     return " ".join(words)

# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))

# st.title("Email/SMS Spam Classifier")

# input_sms = st.text_area("Enter the message")

# if st.button('Predict'):
#     # Preprocess
#     transformed_sms = preprocess_text(input_sms)
#     # Vectorize
#     vector_input = tfidf.transform([transformed_sms])
#     # Predict
#     result = model.predict(vector_input)[0]
#     # Display
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")

import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric characters and stopwords
    words = [ps.stem(word) for word in words if word.isalnum() and word not in stopwords.words('english')]
    
    return " ".join(words)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess
    transformed_sms = preprocess_text(input_sms)
    # Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # Predict
    result = model.predict(vector_input)[0]  # Get the first prediction
    # Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")