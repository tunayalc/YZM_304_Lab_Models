# YZM304 Derin Öğrenme I. Proje Ödevi

**Öğrenci:** Ahmet Tunahan Yalçın  
**Öğrenci No:** 22290665  
**Ders:** YZM304 Derin Öğrenme  

## Introduction

Bu projede, YZM304 laboratuvarında kurulan temel çok katmanlı algılayıcı yapısı `Wine` veri seti üzerinde uygulanmıştır. Veri seti 13 sayısal özellikten ve 3 sınıftan oluşmaktadır. Çalışmanın amacı, laboratuvarda kurulan temel modeli veri ön işleme ve mimari değişikliklerle incelemek, daha sonra aynı yapıyı NumPy, PyTorch ve Scikit-learn ile tekrar kurarak sonuçları karşılaştırmaktır.

Model seçimi validation accuracy ve `n_steps` ölçütlerine göre yapılmıştır. Böylece hem başarı hem de eğitim verimliliği birlikte değerlendirilmiştir.

## Methods

### Veri Seti

- Kaynak: `sklearn.datasets.load_wine`
- Problem tipi: çoklu sınıflandırma
- Özellik sayısı: 13
- Sınıf sayısı: 3
- Toplam örnek sayısı: 178
- Sınıf dağılımı:
  - `class_0 = 59`
  - `class_1 = 71`
  - `class_2 = 48`

Sınıf dağılımı görece dengeli olduğu için ana seçim metriği olarak validation accuracy kullanılmıştır. Bunun yanında precision, recall, F1 ve confusion matrix sonuçları da raporlanmıştır.

### Veri Bölme

Veri `stratified split` ile aşağıdaki şekilde ayrılmıştır:

- Train: %60
- Validation: %20
- Test: %20
- `random_state = 42`

Tüm deneylerde aynı bölme korunmuştur. Bölme özeti şu dosyada tutulur:

- `data/splits/split_manifest.json`

### Ön İşleme

Deneylerde üç farklı kurulum kullanılmıştır:

1. `none`: ham veri
2. `standardize`: train ortalaması ve standart sapması ile standardizasyon
3. `standardize + deeper + L2`: standardizasyon, ikinci gizli katman ve L2 regularizasyonu

Ölçekleme parametreleri yalnızca train setinden hesaplanmış, validation ve test setlerine aynı değerler uygulanmıştır.

### Başlangıç Ayarları

- Global tohum: `42`
- Veri bölme tohumu: `42`
- Tüm backendlerde `shuffle=False`
- Seçilen mimari için aynı başlangıç ağırlıkları kullanılmıştır

Başlangıç ağırlıkları şu dosyalarda tutulur:

- `data/weights/standardized_baseline.npz`
- `data/weights/standardized_baseline.json`

NumPy modelinde ağırlıklar `sqrt(2 / fan_in)` ölçekli normal dağılımla, bias değerleri ise sıfır olarak başlatılmıştır. PyTorch ve Scikit-learn modelleri aynı başlangıç ağırlıklarıyla eşlenmiştir.

### Modeller

#### NumPy MLP

- Gizli katman aktivasyonu: `ReLU`
- Çıkış katmanı: `Softmax`
- Kayıp fonksiyonu: çoklu sınıf `cross-entropy`
- Optimizasyon: `SGD`
- Eğitim tipi: mini-batch
- Sınıf yapısı: `NumpyMLPClassifier`

#### PyTorch MLP

- NumPy modelinin aynı mimari karşılığı kurulmuştur
- Optimizasyon: `torch.optim.SGD`
- Başlangıç ağırlıkları NumPy ile aynıdır

#### Scikit-learn MLPClassifier

- `solver='sgd'`
- `activation='relu'`
- `learning_rate='constant'`
- Batch size, epoch, veri bölmesi ve başlangıç ağırlıkları NumPy modeliyle aynıdır

### Deney Konfigürasyonları

| Model | Ön İşleme | Gizli Katmanlar | Öğrenme Oranı | Epoch | Batch | L2 |
| --- | --- | --- | --- | --- | --- | --- |
| `raw_baseline` | none | `(16,)` | 0.01 | 250 | 16 | 0.0 |
| `standardized_baseline` | standardize | `(16,)` | 0.01 | 250 | 16 | 0.0 |
| `standardized_deeper_l2` | standardize | `(32, 16)` | 0.01 | 300 | 16 | 0.001 |

Seçilen modelin tam mimarisi `13-16-3` şeklindedir.

### Model Seçimi

Model seçimi şu kuralla yapılmıştır:

1. En yüksek `best_val_accuracy`
2. Eşitlik durumunda daha düşük `n_steps`

Bu kurala göre seçilen model `standardized_baseline` olmuştur.

### Kullanılan Kütüphaneler

- NumPy 2.4.3
- Pandas 2.3.3
- Scikit-learn 1.8.0
- PyTorch 2.10.0
- Matplotlib 3.10.8
- Seaborn 0.13.2
- Pytest 8.4.2

### Klasör Yapısı

```text
.
+-- README.md
+-- requirements.txt
+-- assignment
|   `-- yzm304_proje_odevi1_2526.pdf
+-- artifacts
|   +-- plots
|   `-- reports
+-- data
|   +-- splits
|   `-- weights
+-- src
|   +-- data.py
|   +-- metrics.py
|   +-- run_experiments.py
|   `-- models
|       +-- numpy_mlp.py
|       +-- sklearn_mlp.py
|       `-- torch_mlp.py
`-- tests
    +-- test_artifacts.py
    +-- test_data.py
    +-- test_library_wrappers.py
    `-- test_numpy_mlp.py
```

### Tekrar Çalıştırma

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.run_experiments
```

Bu komutlardan sonra aşağıdaki dosyalar oluşur:

- `data/splits/split_manifest.json`
- `data/weights/standardized_baseline.npz`
- `data/weights/standardized_baseline.json`
- `artifacts/reports/custom_experiments.csv`
- `artifacts/reports/custom_experiment_metrics_detailed.csv`
- `artifacts/reports/data_split_summary.json`
- `artifacts/reports/library_comparison.csv`
- `artifacts/reports/summary.json`
- `artifacts/plots/class_distribution.png`
- `artifacts/plots/custom_validation_accuracy.png`
- `artifacts/plots/custom_training_curves.png`
- `artifacts/plots/library_accuracy_comparison.png`
- `artifacts/plots/library_confusion_matrices.png`
- `artifacts/plots/confusion_matrix_raw_baseline.png`
- `artifacts/plots/confusion_matrix_standardized_baseline.png`
- `artifacts/plots/confusion_matrix_standardized_deeper_l2.png`

## Results

### NumPy Deney Sonuçları

| Model | Best Val Accuracy | Final Train Accuracy | Final Val Accuracy | Test Accuracy | Test F1 Macro | n_steps |
| --- | --- | --- | --- | --- | --- | --- |
| `standardized_baseline` | 1.0000 | 1.0000 | 1.0000 | 0.9444 | 0.9453 | 1750 |
| `standardized_deeper_l2` | 1.0000 | 1.0000 | 0.9722 | 0.9722 | 0.9710 | 2100 |
| `raw_baseline` | 0.3889 | 0.4057 | 0.3889 | 0.3889 | 0.1867 | 1750 |

Tablodan görülen temel durumlar şunlardır:

- Ham veriyle kurulan temel model belirgin biçimde underfit olmuştur.
- Standardizasyon tek başına performansı ciddi biçimde artırmıştır.
- Daha derin model testte daha yüksek sonuç verse de seçim kuralına göre daha düşük adımlı temel standart model seçilmiştir.

### Seçilen Model

- Model adı: `standardized_baseline`
- Gizli katman: `(16,)`
- Ön işleme: `standardize`
- Best validation accuracy: `1.0000`
- `n_steps`: `1750`

### Kütüphane Karşılaştırması

| Kütüphane | Accuracy | Precision Macro | Recall Macro | F1 Macro | n_steps |
| --- | --- | --- | --- | --- | --- |
| NumPy | 0.9444 | 0.9505 | 0.9429 | 0.9453 | 1750 |
| PyTorch | 0.9444 | 0.9505 | 0.9429 | 0.9453 | 1750 |
| Scikit-learn | 0.9444 | 0.9505 | 0.9429 | 0.9453 | 1750 |

Üç farklı uygulamanın aynı test sonuçlarını vermesi, kurulan eğitim koşullarının ve başlangıç ağırlıklarının tutarlı olduğunu göstermektedir.

### Karmaşıklık Matrisi

Seçilen model için test karmaşıklık matrisi tüm kütüphanelerde aynıdır:

```text
[[12, 0, 0],
 [ 1,13, 0],
 [ 0, 1, 9]]
```

Bu matris, örneklerin büyük kısmının doğru sınıflandırıldığını ve yalnızca az sayıda karışma olduğunu göstermektedir.

Diğer deneyler için de ayrı confusion matrix görselleri üretilmiştir.

## Discussion

Bu proje, veri ön işlemenin sınıflandırma performansı üzerindeki etkisini açık biçimde göstermektedir. Ham veri ile eğitilen model düşük train ve validation accuracy ile bias ağırlıklı bir underfitting davranışı sergilemiştir. Aynı mimarinin standardizasyon uygulanmış hali ise validation tarafında belirgin bir iyileşme sağlamıştır.

Daha derin ve L2 regularizasyonlu model validation tarafında seçilen modelden daha iyi olmamasına rağmen test setinde biraz daha yüksek accuracy üretmiştir. Bu durum, daha büyük mimarinin bu veri bölmesinde ek temsil gücü sağlayabildiğini ancak seçim kuralında kullanılan `best_val_accuracy + düşük n_steps` yaklaşımının daha yalın bir modeli öne çıkardığını göstermektedir.

PyTorch, NumPy ve Scikit-learn uygulamalarının aynı sonuçları vermesi, laboratuvarda elde yazılan modelin kütüphane tabanlı karşılıklarının da aynı mantıkla çalıştırılabildiğini göstermektedir.

### Gelecek Çalışmalar

İlerleyen aşamalarda daha farklı öğrenme oranları, batch size değerleri ve epoch sayıları denenebilir. Buna ek olarak erken durdurma, farklı aktivasyon fonksiyonları ve daha kapsamlı hiperparametre taramaları ile model davranışı daha ayrıntılı biçimde incelenebilir. Farklı veri setleri üzerinde aynı deney düzeninin uygulanması da elde edilen sonuçların genellenebilirliğini değerlendirmek açısından faydalı olacaktır.
