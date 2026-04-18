# YZM304 Derin Öğrenme II. Proje Ödevi

**Öğrenci:** Ahmet Tunahan Yalçın  
**Öğrenci No:** 22290665  
**Ders:** YZM304 Derin Öğrenme  

## Giriş

Bu projede evrişimli sinir ağları kullanılarak görüntü verisi üzerinde özellik çıkarma ve sınıflandırma çalışması yapılmıştır. Veri seti olarak `CIFAR-10` seçilmiştir. Bunun temel nedeni veri setinin RGB yapıda olması, sınıf dağılımının dengeli olması ve hem açık biçimde yazılan CNN sınıfları hem de literatürde yaygın olarak kullanılan hazır mimariler için uygun bir karşılaştırma zemini sunmasıdır.

Çalışma ödev maddelerine doğrudan karşılık gelecek biçimde beş model üzerinden kurulmuştur:

1. LeNet-5 mantığına benzeyen temel CNN
2. Aynı çekirdek hiperparametrelerini koruyan, batch normalization ve dropout eklenmiş iyileştirilmiş CNN
3. `torchvision.models` içinden alınan hazır `ResNet18` mimarisi
4. Tam CNN modelinden çıkarılan özelliklerle çalışan hibrit `RandomForest` modeli
5. Dördüncü modelle aynı veri üzerinde karşılaştırılan tam CNN modeli

Bu yapı sayesinde hem ders kapsamında kurulan temel ağ hem de daha güçlü hazır mimariler ve hibrit yaklaşım aynı problem üzerinde birlikte incelenmiştir.

## Yöntemler

### Veri Seti

- Veri seti: `torchvision.datasets.CIFAR10`
- Görüntü tipi: RGB
- Giriş boyutu: `32 x 32`
- Kanal sayısı: `3`
- Sınıf sayısı: `10`
- Sınıflar: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

### Veri Bölme

Veri bölme işlemi tekrarlanabilir olması için `seed = 42` ile yapılmıştır.

- Eğitim alt kümesi: `40.000`
- Doğrulama alt kümesi: `10.000`
- Test kümesi: `10.000`

Doğrulama ayrımı `StratifiedShuffleSplit` ile sınıf oranları korunarak oluşturulmuştur. Bölme bilgileri `data/splits/split_manifest.json` içinde saklanmıştır.

### Ön İşleme

Veri seti üzerinde uygulanan ön işleme adımları:

1. Görüntülerin tensöre dönüştürülmesi
2. Kanal bazlı ortalama ve standart sapma değerlerinin yalnızca eğitim alt kümesinden hesaplanması
3. Eğitim, doğrulama ve test bölmelerinin aynı normalizasyon parametreleriyle ölçeklenmesi

Kullanılan normalizasyon değerleri:

- Mean: `[0.4911, 0.4821, 0.4466]`
- Std: `[0.2470, 0.2435, 0.2616]`

Hazır `ResNet18` modeli için aynı veri bölmesi korunmuş, yalnızca girişler `64 x 64` boyutuna yeniden ölçeklenmiştir. Bu seçim, hazır mimariyi makul hesaplama maliyetiyle kullanabilmek için yapılmıştır.

### Model 1: LeNet Benzeri Temel CNN

İlk model ders kapsamındaki LeNet-5 mantığına benzer biçimde açık sınıf olarak yazılmıştır. Kullanılan temel katmanlar:

- `Conv2d(3, 6, kernel_size=5, padding=2)`
- `ReLU`
- `AvgPool2d(2)`
- `Conv2d(6, 16, kernel_size=5)`
- `ReLU`
- `AvgPool2d(2)`
- `Flatten`
- `Linear(16*6*6, 120)`
- `ReLU`
- `Linear(120, 84)`
- `ReLU`
- `Linear(84, 10)`

### Model 2: İyileştirilmiş CNN

İkinci modelde birinci modelin konvolüsyon katmanları korunmuştur. Yalnızca ağı iyileştirmesi beklenen özel katmanlar eklenmiştir:

- `BatchNorm2d`
- `BatchNorm1d`
- `Dropout(p=0.30)`

Bu nedenle ödev maddesindeki "ilk modeldeki katman hiperparametreleri korunmalı, iyileştirme için özel katmanlar eklenmeli" şartı doğrudan karşılanmıştır.

### Model 3: Hazır CNN Mimarisi

Üçüncü model olarak `ResNet18` seçilmiştir.

- Mimari: `ResNet18`
- Başlangıç ayarı: `weights=None`
- Son tam bağlı katman `10` sınıfa göre yeniden yazılmıştır

Hazır modelde önceden eğitilmiş ağırlıklar kullanılmamıştır. Böylece karşılaştırma, dış veriyle gelen bilgiye değil aynı problem üzerinde yapılan eğitime dayandırılmıştır.

### Model 4: Hibrit Model

Dördüncü model hibrit yapıda kurulmuştur. Önce beşinci model olan tam CNN eğitilmiş, daha sonra bu modelin `extract_features` mekanizması kullanılarak özellik dosyaları oluşturulmuştur. Bu özellikler üzerinde klasik makine öğrenmesi modeli eğitilip test edilmiştir.

Üretilen dosyalar:

- `data/features/train_features.npy`
- `data/features/train_labels.npy`
- `data/features/test_features.npy`
- `data/features/test_labels.npy`

Oluşan özellik boyutları:

- Eğitim özellik matrisi: `(40000, 256)`
- Eğitim etiket vektörü: `(40000,)`
- Test özellik matrisi: `(10000, 256)`
- Test etiket vektörü: `(10000,)`

Hibrit sınıflandırıcı olarak `RandomForestClassifier` seçilmiştir. Bu seçim, doğrusal olmayan karar sınırlarını öğrenebilmesi ve çıkarılmış özellik vektörleri üzerinde pratik biçimde uygulanabilmesi nedeniyle yapılmıştır.

### Model 5: Hibrit ile Karşılaştırılan Tam CNN

Beşinci model, hibrit özellik çıkarımını da sağlayan tam CNN mimarisidir.

- `Conv2d(3, 32, kernel_size=3, padding=1)`
- `ReLU`
- `MaxPool2d(2)`
- `Conv2d(32, 64, kernel_size=3, padding=1)`
- `ReLU`
- `MaxPool2d(2)`
- `Conv2d(64, 128, kernel_size=3, padding=1)`
- `ReLU`
- `MaxPool2d(2)`
- `Flatten`
- `Linear(128*4*4, 256)`
- `ReLU`
- `Dropout(0.30)`
- `Linear(256, 10)`

Bu modelin ara temsilleri hibrit modelin girdi kümesini oluşturmuştur. Böylece 4. ve 5. modeller aynı veri üzerinde doğrudan karşılaştırılabilir hale gelmiştir.

### Kayıp Fonksiyonu ve Optimizasyon

CNN tabanlı modellerde kayıp fonksiyonu olarak:

- `nn.CrossEntropyLoss()`

optimizasyon için ise:

- `Adam`

kullanılmıştır.

Seçilen hiperparametreler:

| Model | Epoch | Learning Rate | Batch Size | Weight Decay | Giriş Boyutu |
| --- | ---: | ---: | ---: | ---: | ---: |
| Model 1 | 15 | 0.001 | 128 | 0.0 | 32 |
| Model 2 | 18 | 0.001 | 128 | 0.0001 | 32 |
| Model 3 | 6 | 0.0005 | 64 | 0.0001 | 64 |
| Model 5 | 20 | 0.001 | 128 | 0.0001 | 32 |

`Adam` optimizer seçimi, CIFAR-10 gibi orta ölçekli görüntü problemlerinde kararlı ve hızlı yakınsama sağlaması nedeniyle tercih edilmiştir. `ResNet18` için daha küçük epoch ve orta boy giriş çözünürlüğü kullanılarak hesaplama yükü yönetilebilir tutulmuştur.

### Çalıştırma Adımları

```bash
python -m venv .venv
. .venv/Scripts/activate
python -m pip install -r requirements.txt
python -m pytest -q tests -p no:cacheprovider
python -m src.run_experiments
```

Windows PowerShell kullanan bir ortamda doğrudan script çalıştırmak için:

```powershell
.\scripts\setup_env.ps1
.\scripts\run_pipeline.ps1
```

### Klasör Yapısı

```text
CNN_Project
├── README.md
├── requirements.txt
├── .gitignore
├── assignment
│   └── yzm304_proje2_2526.pdf
├── artifacts
│   ├── plots
│   └── reports
├── data
│   ├── features
│   └── splits
├── scripts
│   ├── run_pipeline.ps1
│   └── setup_env.ps1
├── src
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   ├── reporting.py
│   ├── run_experiments.py
│   ├── training.py
│   └── models
│       ├── hybrid_feature_cnn.py
│       ├── lenet_baseline.py
│       ├── lenet_improved.py
│       └── reference_resnet.py
└── tests
    ├── test_metrics.py
    ├── test_models.py
    └── test_split_utils.py
```

## Sonuçlar

Deneyler CPU üzerinde çalıştırılmıştır. Elde edilen temel sonuçlar aşağıdadır:

| Model | Best Epoch | Test Accuracy | Precision Macro | Recall Macro | F1 Macro | Test Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Model 1 - LeNet Baseline | 13 | 0.6363 | 0.6343 | 0.6363 | 0.6340 | 1.0487 |
| Model 2 - LeNet Improved | 18 | 0.6832 | 0.6839 | 0.6832 | 0.6819 | 0.9254 |
| Model 3 - ResNet18 Reference | 6 | 0.7619 | 0.7616 | 0.7619 | 0.7592 | 0.7595 |
| Model 5 - Full CNN | 19 | 0.7552 | 0.7549 | 0.7552 | 0.7547 | 1.0693 |
| Model 4 - Hybrid RandomForest | - | 0.7521 | 0.7523 | 0.7521 | 0.7520 | - |

Bu sonuçlara göre:

- Temel LeNet benzeri ağ en düşük performansı vermiştir.
- Batch normalization ve dropout eklenmiş ikinci model ilk modele göre belirgin biçimde iyileşmiştir.
- Hazır `ResNet18` modeli en yüksek test doğruluğunu üretmiştir.
- Hibrit model, tam CNN modeliyle oldukça yakın sonuç vermiş ancak az farkla geride kalmıştır.

Üretilen ana çıktı dosyaları:

- `artifacts/reports/experiment_metrics.csv`
- `artifacts/reports/summary.json`
- `artifacts/reports/hybrid_feature_summary.json`
- `artifacts/reports/model_1_lenet_baseline_history.csv`
- `artifacts/reports/model_2_lenet_improved_history.csv`
- `artifacts/reports/model_3_resnet18_reference_history.csv`
- `artifacts/reports/model_5_full_cnn_for_hybrid_comparison_history.csv`
- `artifacts/plots/train_class_distribution.png`
- `artifacts/plots/training_curves_cnn_models.png`
- `artifacts/plots/model_metric_comparison.png`
- `artifacts/plots/confusion_matrix_model_1_lenet_baseline.png`
- `artifacts/plots/confusion_matrix_model_2_lenet_improved.png`
- `artifacts/plots/confusion_matrix_model_3_resnet18_reference.png`
- `artifacts/plots/confusion_matrix_model_4_hybrid_random_forest.png`
- `artifacts/plots/confusion_matrix_model_5_full_cnn_for_hybrid_comparison.png`

## Tartışma

Bu çalışma, CNN tabanlı sınıflandırma problemini tek bir model üzerinden değil, beş farklı çözüm yaklaşımı üzerinden değerlendirmektedir. Böylece yalnızca nihai doğruluk değil, mimari tercihlerin etkisi de gözlenebilmektedir.

Model 1 ders içeriğini doğrudan temsil eden temel referans modeldir. Model 2, aynı çekirdek hiperparametreleri koruyup normalizasyon ve dropout ile iyileştirme getirerek mimari tamamen değiştirilmeden de performans artışı sağlanabildiğini göstermiştir. Gerçekten de accuracy değeri `0.6363` seviyesinden `0.6832` seviyesine yükselmiştir.

Model 3 olan `ResNet18`, hazır mimarilerin temsil gücünün daha yüksek olduğunu göstermiştir. `0.7619` test accuracy değeri ile en iyi sonucu bu model vermiştir. Bu durum, artık daha derin ve daha iyi tasarlanmış mimarilerin aynı veri kümesinde klasik LeNet yapısına göre daha güçlü genelleme sağlayabildiğini göstermektedir.

Model 4 ve Model 5 birlikte değerlendirildiğinde hibrit yaklaşım ile tam CNN arasında çok büyük bir fark oluşmadığı görülmektedir. Tam CNN modeli `0.7552`, hibrit `RandomForest` modeli ise `0.7521` accuracy elde etmiştir. Bu yakınlık, beşinci modelden çıkarılan özelliklerin gerçekten anlamlı bir temsil uzayı oluşturduğunu göstermektedir.

İleri çalışma olarak aşağıdaki adımlar uygulanabilir:

- veri artırma teknikleri eklemek
- `VGG`, `MobileNet` veya `DenseNet` gibi başka hazır mimariler denemek
- hibrit modelde `SVM` veya `Gradient Boosting` kullanarak ek karşılaştırmalar yapmak
- öğrenme oranı zamanlayıcıları ile eğitim dinamiğini iyileştirmek

## Referanslar

1. LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. Gradient-Based Learning Applied to Document Recognition. *Proceedings of the IEEE*, 1998.
2. Krizhevsky, A. Learning Multiple Layers of Features from Tiny Images. Technical Report, 2009.
3. PyTorch Documentation, [`torch.nn.Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
4. PyTorch Documentation, [`torch.nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
5. Torchvision Model Documentation, [`torchvision.models`](https://pytorch.org/vision/stable/models.html)
6. He, K., Zhang, X., Ren, S., Sun, J. Deep Residual Learning for Image Recognition. *CVPR*, 2016.
