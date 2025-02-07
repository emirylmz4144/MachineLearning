import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# NLTK için gerekli verileri indiriyoruz
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


# Metin ön işleme fonksiyonu
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Model ve TF-IDF vectorizer'ını yüklemek
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model5.pkl','rb'))

# Streamlit uygulaması başlığı
st.title("SMS-SPAM DEDEKTÖRÜ")

# Kullanıcıdan metin girişi al
input_sms = st.text_area("Mesajı girin: ")

if st.button('İncele'):
    # 1. Metni ön işleme
    transformed_sms = transform_text(input_sms)

    # 2. Vektörleştirme
    vector_input = tfidf.transform([transformed_sms])

    # 3. Tahmin yapma
    result = model.predict(vector_input)[0]

    # 4. Sonucu gösterme
    if result == 1:
        st.header("Normal mesaj")
    else:
        st.header("Spam mesaj")
