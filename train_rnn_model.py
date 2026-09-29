"""
RNN ile Duygu Dedektifi(Sentiment Analysis)
Problem Tanimi:Bir yorumun olumlu mu olumsuz mu oldugunu anlamak.(classification problem)
    IMDB film yorumlari veri seti ile bir metnin duygusal analizini gerceklestirme. 
    
RNN:Tekrarlayan sinir aglari:Sirali veriler üzerinde calisir.Onceki bilgileri hatirlayarak sonraki tahminleri yapmaya calisirlar.

Veri Seti:IMDB veri seti.Film yorumlari.olumlu-olumsuz.
    -50000 adet film yorumu
    -0 Negatif,1 pozitif

Plan/Program

Gerekli Kurulumlar:pip install tensorflow matplotlib nltk

import libraries
"""
# import libraries
import numpy as np
import nltk   # natural language tool kit
import matplotlib.pyplot as plt # görselleştirme
from nltk.corpus import stopwords # gereksiz kelime listesi
from tensorflow.keras.models import Sequential # base model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.datasets import imdb # veri seti
from tensorflow.keras.preprocessing.sequence import pad_sequences 


# stopWords(gereksiz kelimeler) listesi belirle
nltk.download("stopwords") # nltk icinden stopwords indiriliyor
stop_words=set(stopwords.words("english")) # kucuk ve anlamsiz kelimeler ayiklanacak

# model parametreleri
max_features = 10000 # en cok kullanilan 10 bin kelime
maxlen = 500

# load dataset
(X_train, y_train),(X_test, y_test) = imdb.load_data(num_words = max_features) # train/test ayrilmis sekilde veri gelir.

# örnek veri incelemesi
original_word_index = imdb.get_word_index()

# sayi-kelime donusum sozlugu hazirlama
inv_word_index = { index + 3: word for word, index in original_word_index.items()}
inv_word_index[0] = "<PAD>" # 0: bosluk padding
inv_word_index[1] = "<START>" # 1: cumle baslangici 
inv_word_index[2] = "<UNK>" # 2: bilinmeyen kelime(uknown)
# inv_word_index[3] -> great: 65..... 

# sayi dizisini kelimelere ceviren fonkisyon
def decode_review(encoded_review):
    return " ".join([inv_word_index.get(i, "?") for i in encoded_review])

movie_index = 0
# ilk egitim verisini yazdırma 
print("ilk yorum: (sayi dizisi) ")
print(X_train[movie_index])

print("ilk yorum: (kelimelerle)")
print(decode_review(X_train[movie_index]))


print(f"Label: {"Pozitif" if y_train[movie_index]== 1 else "Negatif"}")

# gerekli sozluklerin olusturulmasi: word to index ve index to word 
word_index = imdb.get_word_index()
index_to_word = {index + 3: word for word, index in word_index.items()} # sayilardan kelimelere gecis
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<UNK>"
word_to_index = {word: index for index, word in index_to_word.items()} # kelimelerden sayilara gecis

# data preprocessing(veri ön işleme)
def preprocess_review(encoded_review):
    #sayilari kelimelere cevir
    words = [index_to_word.get(i, "") for i in encoded_review if i >= 3]

    # sadece harflerden olusan ve stopword olmayan kelimeleri al
    cleaned = [
        word.lower() for word in words if word.isalpha() and word.lower() not in stop_words
    ] 

    # tekrardan temizlenmis metni sayilara cevir
    return [word_to_index.get(word, 2) for word in cleaned]

# veriyi temizle ve sabit uzunlugu pad et
X_train = [preprocess_review(review) for review in X_train]
X_test = [preprocess_review(review) for review in X_test]

# pad sequence
"""
merhaba bugun hava cok guzel
merhaba, naber , 0, 0,  0
"""
X_train = pad_sequences(X_train, maxlen = maxlen)
X_test = pad_sequences(X_test, maxlen = maxlen)

# RNN modeli oluşturma
model = Sequential() # base model: katmanlari sirali olarak eklemek icin

# embedding katmani: kelime indexlerini 32 boyutlu bir vektore donusturur.
model.add(Embedding(input_dim = max_features, output_dim = 32, input_length = maxlen))

# simpleRNN katmani: metni sirayla isler ve baglam iliskisini ogrenir
model.add(SimpleRNN(units = 32)) # cell(hücre)(nöron) sayisi

# output katmani: ikili siniflandirma(binary classification: sigmoid, 1 hucre )
"""
sigmoid 0 ile 1 arasindaki olasiligi verir.
negatif -> 0.7
 pozitif ->0.3
"""

model.add(Dense(1, activation = "sigmoid"))

# model compile
model.compile(
    optimizer = "adam", # agirlik guncellemesi icin kullanilan algoritma
    loss = "binary_crossentropy" , # kayip fonksiyonu 
    metrics = ["accuracy"] # degerlendirme metrigi
)

print(model.summary())

# training(modelin eğitilmesi)
history = model.fit(
    X_train, y_train, # girdi ve cikti verileri
    epochs = 2, # egitim tekrar sayisi yani tum veriyi 2 kere egit
    batch_size = 64, # torba,ayni anda islenecek ornek sayisi yani 64 lu paketler halinde isle
    validation_split = 0.2 # %20 yi dogrulama icin ayir
)

# model evaluation(test işlemi,gorsellestirme ,kayip-dogruluk grafikleri)
def plot_history(hist):
    plt.figure(figsize = (12,4))

    # accuracy
    plt.subplot(1, 2, 1)
    plt.plot(hist.history["accuracy"], label = "Training")
    plt.plot(hist.history["val_accuracy"], label = "Validation")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend() # etiketleri gorunur hale getirir

    # loss(kayip grafigi)
    plt.subplot(1, 2, 2)
    plt.plot(hist.history["loss"], label = "Training")
    plt.plot(hist.history["val_loss"], label = "Validation")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history)

# test verisiyle modeli degerlendirme
test_loss, test_acc = model.evaluate(X_test, y_test) # test
print(f"Test: {test_acc:.2f}")

# egitilen modelin kaydini yapalim
model.save("rnn_duygu_model.h5")
print(f"Model basariyla kaydedildi: rnn_duygu_model.h5")