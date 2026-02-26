# 📅 Day 5 — Veri Doldurma | NLP | Normalizasyon & PCA | Derin Öğrenme | Bilgisayarlı Görü

Bu günde eksik veri doldurma teknikleri, doğal dil işleme (spaCy), normalizasyon ve boyut azaltma (PCA), TensorFlow/Keras ile yapay sinir ağları, erken durdurma, PyTorch ile MLP ve CNN kullanarak görüntü sınıflandırma konuları çalışıldı.

---

## 📁 Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `DataImputation.ipynb` | Eksik veri doldurma yöntemleri (SimpleImputer, KNNImputer, IterativeImputer, miceforest) |
| `Normalization-PCA.ipynb` | Veri normalizasyonu, PCA ile boyut azaltma, Keras ile DL (diabetes + house price + cars) |
| `denemeNLP.ipynb` | spaCy ile doğal dil işleme — benzerlik, NER, pipeline analizi |
| `ComputerVisionWithDL.ipynb` | CNN ile MNIST el yazısı rakam tanıma |
| `ExampleData.csv` | Eksik değer içeren örnek Height/YOE/Salary verisi |
| `PastHires.csv` | İşe alım geçmişi verisi — Years Experience, Education, Hired |
| `pima-indians-diabetes.csv` | Pima yerlileri diyabet veri seti (768 kayıt, 8 özellik) |
| `kc_house.pkl` | King County ev fiyatı verisi (pickle formatı) |
| `cars.xls` | Araba fiyat verisi — Deep Learning regression için |

---

## 🧩 1. Data Imputation — Eksik Veri Doldurma (`DataImputation.ipynb`)

Eksik verilerle başa çıkmanın birden fazla yolu vardır.

### Yöntemler

| Yöntem | Sınıf | Strateji |
|--------|-------|----------|
| Ortalama ile doldurma | `SimpleImputer(strategy='mean')` | Basit istatistik |
| En yakın komşu | `KNNImputer()` | K-NN mesafesine göre |
| Regresyon ile | `IterativeImputer(estimator=LinearRegression())` | Diğer sütunları kullanarak tahmini |
| MICE Forest | `miceforest.ImputationKernel()` | Çok değişkenli zincirleme denklemler |
| İleri/geri doldurma | `df.ffill()` / `df.bfill()` | Zaman serisi için sıralı doldurma |

```python
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import LinearRegression

# Ortalama ile doldur
imp = SimpleImputer(strategy='mean')
dolu = imp.fit_transform(df)

# KNN ile doldur
knn = KNNImputer()
dolu = knn.fit_transform(df)

# Regresyon ile doldur
lr = LinearRegression()
imp = IterativeImputer(estimator=lr)
dolu = imp.fit_transform(df)

# MICE Forest
import miceforest as mf
imp = mf.ImputationKernel(df)
```

---

## 📐 2. Normalizasyon & PCA (`Normalization-PCA.ipynb`)

### Normalizasyon vs. Ölçeklendirme

```python
from sklearn.preprocessing import normalize, scale

# Normalize: her satırın uzunluğunu 1 yapar (L2 norm)
yenidf = normalize(df)

# Scale: sütun bazlı z-score standardizasyonu (mean=0, std=1)
scaled_data = scale(df)
```

### PCA — Boyut Azaltma

Veri setinin boyutunu koruyarak bilgi kaybını minimize eder.

```python
from sklearn.decomposition import PCA

pca = PCA(3)            # 3 bileşene indir
data2 = pca.fit_transform(df)
print(data2.shape)      # (n_samples, 3)
```

> PastHires.csv içindeki kategorik değişkenler önce `map()` ve `replace()` ile sayıya çevrildi: Y/N → 1/0, BS/MS/PhD → 1/2/3

---

## 🧠 3. Derin Öğrenme — TensorFlow/Keras (`Normalization-PCA.ipynb`)

### 3.1 İkili Sınıflandırma — Diabetes Dataset

Pima yerlileri diyabet veri seti üzerinde çok katmanlı sinir ağı.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(80,  activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(64,  activation='relu'))
model.add(Dense(30,  activation='relu'))
model.add(Dense(8,   activation='relu'))
model.add(Dense(1,   activation='sigmoid'))   # ikili çıktı

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(x, y, batch_size=32, validation_split=0.10, epochs=100)
```

> ⚠️ StandardScaler uygulandıktan sonra doğruluk önemli ölçüde arttı.

### 3.2 Regresyon — King County Ev Fiyatı

```python
# Doğru train/test split ile data leakage önlendi
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)   # fit sadece train'de
x_test  = scaler.transform(x_test)        # test'e sadece transform

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=120, batch_size=64)

from sklearn.metrics import r2_score
print(r2_score(y_test, model.predict(x_test)))
```

### 3.3 Early Stopping — Cars Dataset

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    callbacks=[early_stop],
    epochs=120
)
```

> Early Stopping, val_loss 10 epoch boyunca iyileşmezse eğitimi durdurur — overfitting önler ve zaman tasarrufu sağlar.

---

## 🔥 4. PyTorch ile MLP (`Normalization-PCA.ipynb`)

TensorFlow'a ek olarak aynı problem PyTorch ile de çözüldü.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 80)
        self.fc2 = nn.Linear(80, 120)
        self.fc3 = nn.Linear(120, 64)
        self.fc4 = nn.Linear(64, 30)
        self.fc5 = nn.Linear(30, 8)
        self.fc6 = nn.Linear(8, 8)
        self.fc7 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        ...
        return self.fc7(x)

model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
```

> Manuel early stopping döngüsü ile val_loss takip edildi; patience=10 ayarıyla eğitim erken sonlandırıldı.

---

## 💬 5. NLP ile spaCy (`denemeNLP.ipynb`)

### Kurulum ve Model Yükleme

```bash
pip install spacy
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

### Cümle Benzerliği (Semantic Similarity)

```python
import spacy
nlp = spacy.load("en_core_web_md")   # md model vektör içerir

doc1 = nlp("I like salty fries and hamburgers")
doc2 = nlp("Fast food tastes very good")
print(doc1.similarity(doc2))    # 0.0–1.0 arası skor

# Token/Span benzerliği
french_fries = doc1[2:4]
burgers = doc1[5]
print(french_fries.similarity(burgers))
```

### Pipeline Analizi

```python
# Boş model
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")
nlp.analyze_pipes()

# Dolu model
nlp2 = spacy.load("en_core_web_sm")
nlp2.analyze_pipes()   # tokenizer, tagger, parser, ner...
```

### Wikipedia API Entegrasyonu

```python
import wikipediaapi

wiki = wikipediaapi.Wikipedia(user_agent="MyAgent", language='tr')
page = wiki.page("Amerika_Birleşik_Devletleri")
print(page.summary[:500])
```

---

## 👁️ 6. Computer Vision — CNN ile MNIST (`ComputerVisionWithDL.ipynb`)

El yazısı rakam sınıflandırma: 60.000 eğitim, 10.000 test görüntüsü (28×28 piksel).

### Model Mimarisi

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Reshape, Conv2D, MaxPooling2D, Flatten, Dense

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalizasyon: piksel değerlerini 0-1 arasına çek
train_images = train_images / 255.0
test_images  = test_images / 255.0

# CNN Modeli
model = Sequential([
    InputLayer(input_shape=(28, 28)),
    Reshape((28, 28, 1)),

    Conv2D(32, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(10)   # 10 sınıf (0-9), logits
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.1)
```

### Tek Görüntü Tahmini

```python
import numpy as np

index = 999
img_batch = train_images[index].reshape(1, 28, 28)
prediction = model.predict(img_batch)
predicted_class = np.argmax(prediction, axis=1)[0]

print(f"Tahmin: {predicted_class}, Gerçek: {train_labels[index]}")
```

### CNN Katman Rolleri

| Katman | Görev |
|--------|-------|
| `Conv2D` | Görüntüden özellik (kenar, şekil) çıkarır |
| `MaxPooling2D` | Boyut azaltma, en önemli özelliği alır |
| `Flatten` | 2D haritayı 1D vektöre dönüştürür |
| `Dense` | Tam bağlı katman — sınıflandırma yapar |

---

## 📦 Kullanılan Kütüphaneler

![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=flat&logo=spacy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white)

| Kütüphane | Kullanım Amacı |
|-----------|----------------|
| `sklearn.impute` | Eksik veri doldurma |
| `miceforest` | MICE algoritması ile imputation |
| `sklearn.preprocessing` | normalize, scale, StandardScaler |
| `sklearn.decomposition` | PCA boyut azaltma |
| `tensorflow.keras` | DL model oluşturma (Sequential, Dense, Conv2D) |
| `torch` / `torch.nn` | PyTorch MLP implementasyonu |
| `spacy` | Doğal dil işleme, benzerlik, NER |
| `wikipediaapi` | Wikipedia içeriği çekme |
| `matplotlib` | Eğitim grafiği görselleştirme |

---

## 💡 Önemli Notlar

- **Data Leakage**: `StandardScaler.fit()` sadece train verisi üzerinde çalıştırılmalı — test verisine sadece `transform()` uygulanır.
- **Early Stopping**: `patience` parametresi ne kadar büyük olursa model o kadar uzun eğitilir ama overfitting riski artar.
- **CNN vs MLP**: CNN görüntülerde uzamsal özellikleri öğrenirken MLP tüm pikselleri düz vektör olarak görür — bu yüzden CNN görüntülerde daha başarılıdır.
- **spaCy Model Seçimi**: `en_core_web_sm` NER ve parsing için yeterli; benzerlik için vektör içeren `en_core_web_md` gereklidir.
- **PCA**: Boyut azaltmadan önce standartlaştırma yapılmalıdır, aksi hâlde büyük ölçekli sütunlar baskın bileşen olur.
