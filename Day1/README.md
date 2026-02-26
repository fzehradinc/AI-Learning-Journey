# 📅 Day 1 — Python'a Giriş & Pandas Temelleri

Bu klasör, yapay zeka öğrenme yolculuğumun **1. gününe** ait çalışmaları içermektedir.  
Ana konular: Python temel sözdizimi, Pandas ile veri okuma/analiz, Excel/CSV dosyaları ve görselleştirmeye giriş.

---

## 📁 Dosya Açıklamaları

| Dosya | Açıklama |
|---|---|
| `Day1.ipynb` | Günün ana çalışma defteri — Python ve Pandas alıştırmaları |
| `Intro to Python.ipynb` | Python'a giriş: değişkenler, listeler, sözlükler, döngüler, koşullar |
| `train.csv` | Titanic veri seti — Pandas ile veri okuma ve analiz pratiği |
| `test_pandas.csv` | Pandas okuma testleri için küçük örnek CSV |
| `test_pandas_no_header.csv` | Başlıksız CSV okuma pratiği |
| `test_pandas.db` | SQLite veritabanı okuma denemesi |
| `test_pandas.xlsm` | Makro içeren Excel dosyası okuma testi |
| `test_sheets.xlsx` | Çok sayfalı Excel dosyası okuma pratiği |
| `istatistik.xlsx` | `df.describe()` çıktısından üretilen istatistik tablosu |
| `excel-comp-data.xlsx` | Excel karşılaştırma veri seti |
| `sales-estimate.xlsx` | Satış tahmini Excel verisi |
| `sales-funnel.xlsx` | Satış hunisi Excel verisi |
| `Courses.json` | JSON formatında kurs ve ücret verisi |
| `catalog.xml` | XML formatında bitki kataloğu |
| `data.xml` | XML formatında öğrenci verisi |
| `robot.mp3` | gTTS kütüphanesi ile Türkçe üretilmiş ses dosyası |
| `robot_en.mp3` | gTTS kütüphanesi ile İngilizce üretilmiş ses dosyası |
| `.ipynb_checkpoints/` | Jupyter otomatik kayıt klasörü |

---

## 🧠 Öğrenilen Temel Kavramlar

### Python Temelleri
- `print()`, değişken tanımlama (`int`, `float`, `str`)
- Liste işlemleri: `append()`, `extend()`, `len()`, indeksleme
- Sözlük (`dict`) kullanımı
- Koşullu ifadeler: `if / elif / else`
- Döngüler: `for` döngüsü ve **list comprehension**
- `input()` ile kullanıcıdan veri alma

### Pandas
- `pd.read_csv()` ile CSV okuma
- `df.head()`, `df.tail()`, `df.shape`, `df.describe()`
- `df.corr()` ile korelasyon analizi
- `df.to_excel()` ile Excel çıktısı alma

### Veri Görselleştirme
- `seaborn.pairplot()` ile çok değişkenli görselleştirme
- `hue` parametresi ile sınıf bazlı renklendirme

### Ses Üretimi (gTTS)
- `gTTS` kütüphanesi ile metni sese çevirme
- Türkçe ve İngilizce ses dosyası kaydetme
- `IPython.display.Audio` ile notebook içinde oynatma

### Veri Formatları
- CSV, Excel (`.xlsx`, `.xlsm`), JSON, XML, SQLite okuma temelleri

---

## 📝 Notlar

- Titanic veri seti, Pandas'ın temel fonksiyonlarını pekiştirmek için kullanıldı.
- `seaborn` pairplot ile `Age`, `Fare`, `Pclass`, `SibSp` değişkenleri arasındaki ilişkiler incelendi.
- gTTS internet bağlantısı gerektirmektedir.

---

## 🛠️ Kullanılan Kütüphaneler

```
pandas
seaborn
matplotlib
gtts
IPython
```
