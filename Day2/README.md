# 📅 Day 2 — İleri Pandas, EDA, Görselleştirme & Gerçek Dünya Verisi

Bu klasör, yapay zeka öğrenme yolculuğumun **2. gününe** ait çalışmaları içermektedir.  
İlk günün temellerinin üstüne inşa edilerek; **keşifçi veri analizi (EDA)**, **özellik mühendisliği**, **finansal veri çekme** ve **Google Trends analizi** konularına girildi. Tüm çalışmalar gerçek dünya veri setleri üzerinde uygulandı.

---

## 📁 Dosya Açıklamaları

| Dosya | Açıklama |
|---|---|
| `Day2_Continue.ipynb` | Günün ana çalışma defteri — tüm konular bu notebook'ta uygulandı |
| `train.csv` | Titanic veri seti — ileri Pandas operasyonları için kullanıldı |
| `supermarket.csv` | Myanmar süpermarket satış verisi — EDA ve görselleştirme uygulandı |
| `combine.csv` | NFL Combine verisi — çoklu veri seti çalışması |
| `corona_dat.csv` | COVID-19 küresel vaka verisi — zaman serisi formatında |
| `Methane_final.csv` | Küresel metan emisyon verisi |
| `fe.csv` | Özellik mühendisliği sonrası üretilen supermarket verisi |
| `istatistik.xlsx` | `df.describe()` çıktısından üretilen istatistik özeti |
| `supermarket.png` | Seaborn ile üretilen görselleştirme çıktısı |
| `new.txt` | Notlar |

---

## 🧠 Öğrenilen Temel Kavramlar

### 🐼 İleri Pandas

| Fonksiyon | Açıklama |
|---|---|
| `df.sample(n)` | Rastgele n satır göster |
| `df.isnull().sum()` | Sütun bazında eksik veri sayısı |
| `df.loc[rows, cols]` | Etiket tabanlı indeksleme |
| `df[df['Age'] > 25]` | Koşullu filtreleme |
| `df.columns.str.upper()` | Sütun isimlerini büyük harfe çevir |
| `df['col'].std()` / `.mean()` | Standart sapma ve ortalama |
| `df.info()` | Veri tipi ve memory özeti |
| `pd.to_datetime()` | Sütunu datetime formatına çevir |

### 🛠️ Özellik Mühendisliği (Feature Engineering)
- `dt.day`, `dt.year`, `dt.month_name()`, `dt.day_name()` ile tarih sütunundan yeni özellikler türetme
- `df.to_csv('fe.csv')` ile işlenmiş verinin kaydedilmesi

### 📊 Görselleştirme (Matplotlib & Seaborn)

```python
# NumPy ile veri üretme
x = np.arange(20)
y = np.random.normal(10, 1, 20)

# Matplotlib grafikleri
plt.plot(x, y, label='Y')   # çizgi grafik
plt.scatter(x, y)            # saçılım grafik

# Seaborn grafikleri
sns.countplot(x='month_name', hue='month_name', data=df)
sns.countplot(x='Product line', hue='Gender', data=df)
sns.boxplot(x='Branch', y='Total', data=df)   # outlier tespiti
sns.kdeplot(x=df['Total'])                    # yoğunluk dağılımı
```

### 📈 Finansal Veri Analizi (yfinance)
```python
import yfinance as yf

# ABD hisseleri
df = yf.download(['TSLA', 'AAPL'], start='2023-1-1', end='2025-12-28')
df['Close'].plot()

# Borsa İstanbul hisseleri
df = yf.download(['QNBTR.IS', 'ASELS.IS', 'GARAN.IS'], start='2023-1-1', end='2025-12-28')
```

### 🔍 Google Trends Analizi (pytrends)
```python
from pytrends.request import TrendReq
pt = TrendReq()
pt.build_payload(['Keyword1', 'Keyword2'], timeframe='today 5-y')
df = pt.interest_over_time()
df.plot()
```

---

## 📊 Üzerinde Çalışılan Veri Setleri

| Veri Seti | Konu | Boyut |
|---|---|---|
| `train.csv` | Titanic yolcu hayatta kalma verisi | 891 satır |
| `supermarket.csv` | Myanmar 3 şehir süpermarket satışları | 1000 satır, 17 sütun |
| `combine.csv` | NFL oyuncu fiziksel ölçüm verisi | ~4500 satır |
| `corona_dat.csv` | COVID-19 ülke bazlı günlük vaka sayısı | 180+ ülke |
| `Methane_final.csv` | Küresel metan emisyon ölçümleri | — |

---

## 📌 Önemli Notlar

- **BoxPlot**, outlier (aykırı değer) tespitinde kullanılan temel araçtır.
- **KDE plot** (Kernel Density Estimation), histogram'ın sürekli hale getirilmiş versiyonudur.
- `pickle` → **turşu** 🥒 — Python nesnelerini diske kaydetmek için kullanılan format (kavramsal not).
- `yfinance` ile gerçek zamanlı hisse senedi verisi çekilebilir; Borsa İstanbul için `.IS` eki kullanılır.
- `pytrends` ile Google arama trendleri API'siz olarak çekilebilir.

---

## 🛠️ Kullanılan Kütüphaneler

```
pandas
numpy
matplotlib
seaborn
yfinance
pytrends
openpyxl
```
