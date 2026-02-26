# 📅 Day 3 — Regresyon & Makine Öğrenmesine Giriş

Bu klasör, yapay zeka öğrenme yolculuğumun **3. gününe** ait çalışmaları içermektedir.  
Makine öğrenmesinin temel taşı olan **regresyon algoritmaları** gerçek dünya veri setleri üzerinde uygulandı. Uçtan uca bir ML pipeline kuruldu: EDA → Feature Engineering → Outlier Temizliği → Model Karşılaştırma → Sonuç.

---

## 📁 Dosya Açıklamaları

| Dosya | Açıklama |
|---|---|
| `House Price Prediction.ipynb` | **Ana proje** — King County ev fiyat tahmini, LinearRegression + EDA + Feature Engineering |
| `Ev-fiyatlari-tahmini.ipynb` | **Gelişmiş proje** — 10 farklı model karşılaştırması, ExtraTrees, Target Encoding |
| `Day3-Regression.ipynb` | Regresyon algoritmalarının temel uygulamaları |
| `kc_house_data.csv` | ABD Washington/King County 21.613 ev satış verisi (Kaggle) |
| `cars.csv` | Araç fiyat tahmini için ikinci el araba verisi |
| `cars.db` | Araç verisinin SQLite formatı |
| `cars.json` | Araç verisinin JSON formatı |
| `cars.xls` | Araç verisinin Excel formatı |

---

## 🏠 Proje: King County Ev Fiyat Tahmini

> **Hedef:** R² > 0.80 ve RMSE < 100.000$ olan bir regresyon modeli geliştirmek

### Veri Seti Özellikleri

| Sütun | Açıklama |
|---|---|
| `price` | 🎯 Hedef değişken — ev satış fiyatı |
| `sqft_living` | Evin iç alan metrekaresi |
| `grade` | King County puanlama sistemi (1-13) |
| `sqft_above` | Bodrum katı hariç alan |
| `bathrooms` | Banyo sayısı |
| `view` | Manzara puanı |
| `sqft_living15` | 2015'teki iç alan (komşu ortalaması) |
| `waterfront` | Deniz/göl manzarası var mı? |
| `yr_built` | İnşaat yılı |
| `zipcode` | Posta kodu |
| `lat / long` | Coğrafi koordinatlar |

---

## 🧠 Öğrenilen Temel Kavramlar

### 📐 Regresyon Algoritmaları

| Model | Tür | Özellik |
|---|---|---|
| `LinearRegression` | Doğrusal | Baseline model |
| `Ridge` | Doğrusal + L2 | Aşırı öğrenmeyi önler |
| `Lasso` | Doğrusal + L1 | Özellik seçimi yapar |
| `ElasticNet` | L1 + L2 karışımı | Ridge + Lasso kombinasyonu |
| `KNeighborsRegressor` | Instance-based | En yakın komşu tahmini |
| `SVR` | Kernel tabanlı | Yüksek boyutlu uzayda çalışır |
| `DecisionTreeRegressor` | Ağaç | Yorumlanabilir, overfitting'e eğilimli |
| `RandomForestRegressor` | Ensemble | Birçok ağacın ortalaması |
| `ExtraTreesRegressor` | Ensemble | Random Forest'tan daha hızlı ve güçlü |
| `GradientBoostingRegressor` | Boosting | Hataları sırayla düzeltir |

### 📏 Model Değerlendirme Metrikleri

```python
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

r2   = r2_score(y_test, y_pred)          # 1'e ne kadar yakınsa o kadar iyi
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Dolar cinsinden hata
```

| Metrik | Açıklama | Hedef |
|---|---|---|
| **R²** | Modelin varyansı açıklama oranı | > 0.80 |
| **RMSE** | Tahmin hatasının karekökü | < 100.000$ |

### 🛠️ Feature Engineering (Özellik Mühendisliği)

```python
# Yaş hesaplama
df['house_age'] = 2015 - df['yr_built']

# Renovasyon bayrağı
df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)

# Tarihten özellik üretme
df['year']  = pd.to_datetime(df['date']).dt.year
df['month'] = pd.to_datetime(df['date']).dt.month

# Oran özellikleri
df['living_ratio']  = df['sqft_living'] / (df['sqft_lot'] + 1)
df['above_ratio']   = df['sqft_above'] / (df['sqft_living'] + 1)
df['sqft_per_room'] = df['sqft_living'] / (df['bedrooms'] + 1)

# Polinom özellikler
df['bedrooms']    = df['bedrooms'] ** 2
df['sqft_living'] = df['sqft_living'] ** 2
```

### 🧹 Outlier Temizliği

```python
# Kantil tabanlı temizlik (%1-%99 aralığı koru)
Q1 = df['price'].quantile(0.01)
Q3 = df['price'].quantile(0.99)
df_clean = df[(df['price'] >= Q1) & (df['price'] <= Q3)]
```

### ⚖️ Normalizasyon

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform
X_test_scaled  = scaler.transform(X_test)        # sadece transform!
```

> ⚠️ **Kritik Kural:** `fit_transform` sadece train setine uygulanır. Test seti sadece `transform` edilir — veri sızıntısı (data leakage) önlenir.

### 🏷️ Target Encoding (Zipcode için)

```python
# Leakage'siz Target Encoding
train_zip_mean = df_train.groupby('zipcode')['price'].mean()
X_train['zipcode_price_mean'] = X_train['zipcode'].map(train_zip_mean)
X_test['zipcode_price_mean']  = X_test['zipcode'].map(train_zip_mean)
# Görülmemiş zipcode → global ortalama
X_test['zipcode_price_mean'].fillna(y_train.mean(), inplace=True)
```

### 📊 EDA Görselleştirmeleri

```python
# Korelasyon ısı haritası
plt.figure(figsize=(20, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True)

# En çok etkileyen özellikler
abs(df.corr(numeric_only=True)['price']).sort_values(ascending=False)

# Outlier tespiti
sns.boxplot(x=df['bedrooms'])

# Kalıntı dağılımı
sns.kdeplot(residuals['price'], fill=True)
```

### 🧪 ML Pipeline Özeti

```python
# 1. Veri oku
df = pd.read_csv('kc_house_data.csv')

# 2. Feature Engineering
# ... (yukarıdaki dönüşümler)

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Normalize (sadece doğrusal modeller için)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

# 5. Model eğit ve değerlendir
model = ExtraTreesRegressor(n_estimators=600, n_jobs=-1)
model.fit(X_train, y_train)
r2_score(y_test, model.predict(X_test))
```

---

## 📌 Önemli Notlar

- **Correlation ≠ Causation** (Korelasyon nedensellik değildir)
- `sqft_basement` = `sqft_living` − `sqft_above` → redundant, çıkarılabilir
- **ExtraTrees**, Random Forest'a göre daha hızlı ve çoğu durumda daha iyi sonuç verir
- `zipcode` sayısal gibi görünse de kategorik bir değişkendir → `pd.get_dummies()` veya Target Encoding uygulanmalı
- `yellowbrick` kütüphanesi residual plot ve model görselleştirme için kullanılır

---

## 🛠️ Kullanılan Kütüphaneler

```
pandas
numpy
matplotlib
seaborn
scikit-learn
yellowbrick
```
