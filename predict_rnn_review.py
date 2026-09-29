"""
egitilmis rnn modelini kullanarak kullanici yorumlarini analiz edelim
"""

import numpy as np 
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

# Model Parametreleri
max_features = 10000 # egitim sirasinda kullanilan maksimum kelime sayisi
maxlen = 500 # rnn modelinin bekledigi  sabit uzunluk input len

# stopwords kurtulma ve sozlukleri hazirlama
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# imdb verisetinden  kelime -> index sozlugu alalim
word_index = imdb.get_word_index()

# sayi -> kelime sozlugu olusturma
index_to_word = {index + 3: word for word, index in word_index.items()}
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<UNK>"

# kelime -> sayi donusumu icin ters sozluk
word_to_index = {word: index for index, word in index_to_word.items()}

# egitim modelini yukle
model = load_model("rnn_duygu_model.h5")
print("Model basariyla yuklendi.")

# Tahmin yapma fonksiyonu
def predict_review(text):
    """
        kullanicidan gelen metni temizler,modele uygun hale getirir,tahmin sonucunu yazdirir.
    """

    # Yorumu kucuk harfli kelime listesine cevir
    words = text_to_word_sequence(text) # orn: This movie is great -> ["this","movie","is","great"]

    # stopwprds cikarma ve sadece kelimeleri alma
    cleaned = [
        word.lower() for word in words if word.isalpha() and word.lower() not in stop_words
    ]

    # her kelime egitilen sozlukten sayiya cevrilir
    encoded = [word_to_index.get(word, 2) for word in cleaned] # 2 = <UNK>

    # modelin bekledigi sabit uzunluga padding yapiyoruz
    padded = pad_sequences([encoded], maxlen = maxlen)

    # Tahmin yapma islemi (prediction -> 0 ile 1 arasinda bir sonuc dondurur)
    prediction = model.predict(padded)[0][0]

    print(f"Pozitif Tahmin Olasiligi: {prediction:.4f}")
    if prediction > 0.5:
        print("Pozitif")
    else: print("Negatif")

# Kullanici girisi alalim ve tahmin yapalim
# Konsol uzerinden kullanicidan yorum alalim
user_review = input("Bir Film Yorumu Girin: ")
predict_review(user_review)
