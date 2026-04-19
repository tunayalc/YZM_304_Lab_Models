# YZM304 Derin Öğrenme Dersi Proje 2

**Öğrenci:** Ahmet Tunahan Yalçın  
**Öğrenci No:** 22290665  
**Bölüm:** Ankara Üniversitesi Yapay Zeka ve Veri Mühendisliği  
**Dönem:** 2025-2026 Bahar  

Bu depo, CNN kullanarak özellik çıkarma ve sınıflandırma yapılan proje ödevinin teslime hazır halidir. Çalışma doğrudan ödev metnindeki maddelere göre kurulmuştur: iki açık yazılmış CNN sınıfı, bir hazır literatür mimarisi, bir hibrit model ve aynı veri uzayında karşılaştırılan tam CNN modeli birlikte eğitilip test edilmiştir.

## Giriş

Bu projede görüntü verisi üzerinde evrişimli sinir ağları ile sınıflandırma ve özellik çıkarma çalışması yapılmıştır. Veri seti olarak `MNIST` seçilmiştir. Bu seçim ödev metniyle doğrudan uyumludur; çünkü metin benchmark veri setlerinden birinin kullanılabileceğini açıkça söylemektedir. `MNIST`, tek kanallı ve dengeli bir veri seti olduğu için hem LeNet-5 benzeri açık CNN yapılarını hem de hazır CNN mimarilerini temiz biçimde karşılaştırmaya uygun bir taban sunmaktadır.

Bu çalışmada beş model birlikte ele alınmıştır:

1. LeNet-5 benzeri temel CNN
2. Aynı çekirdek katman hiperparametrelerini koruyan, batch normalization ve dropout eklenmiş iyileştirilmiş CNN
3. Hazır mimari olarak kullanılan `ResNet18`
4. Tam CNN'den çıkarılan özelliklerle eğitilen hibrit `LinearSVC`
5. Hibrit model ile aynı veri uzayında karşılaştırılan tam CNN modeli

Böylece hem ders kapsamındaki temel CNN mantığı hem de daha güçlü temsil uzayı üreten hazır ve hibrit yaklaşımlar aynı problem üzerinde birlikte değerlendirilmiş oldu.

## Yöntem

### Veri seti ve bölme

- Veri seti: `torchvision.datasets.MNIST`
- Görüntü tipi: gri seviye
- Standart giriş boyutu: `1 x 32 x 32`
- Sınıf sayısı: `10`
- Sınıflar: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`

Veri bölmesi `seed = 42` ile ve `StratifiedShuffleSplit` kullanılarak yapılmıştır:

- Eğitim: `50.000`
- Doğrulama: `10.000`
- Test: `10.000`

Bölme bilgileri `data/splits/split_manifest.json` dosyasına yazdırılmıştır.

### Ön işleme

Uygulanan ön işleme adımları sadece ödev metninin izin verdiği çekirdek akışta tutulmuştur:

1. Görüntülerin tensöre çevrilmesi
2. Eğitim alt kümesinden mean/std hesaplanması
3. Normalizasyon
4. Boyutun `32 x 32` olacak şekilde eşlenmesi

Eğitim alt kümesinden hesaplanan normalizasyon değerleri:

- Mean: `[0.13101358806265007]`
- Std: `[0.2894179023904892]`

Model 1, Model 2 ve Model 5 için giriş `1 x 32 x 32` tutulmuştur.  
Model 3 için tek kanallı görüntü 3 kanala kopyalanmış ve `224 x 224` boyutuna getirilmiştir. Bu tercih, hazır `ResNet18` mimarisini ödev metninden sapmadan kullanabilmek için yapılmıştır.

Ödev metninde açıkça yer almayan ek veri çoğaltma adımları kullanılmamıştır.

### Model 1: LeNet benzeri temel CNN

İlk model açık sınıf olarak yazılmıştır. Kullanılan katman yapısı:

- `Conv2d(1, 6, kernel_size=5, padding=2)`
- `ReLU`
- `AvgPool2d(2)`
- `Conv2d(6, 16, kernel_size=5)`
- `ReLU`
- `AvgPool2d(2)`
- `Flatten`
- `Linear(16 * 6 * 6, 120)`
- `ReLU`
- `Linear(120, 84)`
- `ReLU`
- `Linear(84, 10)`

Bu model doğrudan temel referans model olarak kullanılmıştır.

### Model 2: İyileştirilmiş CNN

İkinci modelde ilk modelin çekirdek konvolüsyon yapısı korunmuştur. Ağları iyileştirmesi beklenen özel katmanlar uygun yerlere eklenmiştir:

- `BatchNorm2d`
- `BatchNorm1d`
- `Dropout`

Bu nedenle ödev metnindeki "ilk modeldeki katmanların hiperparametreleri aynı kalacak, sadece batch normalization ya da dropout gibi özel katmanlar eklenebilir" koşulu doğrudan sağlanmıştır.

### Model 3: Hazır CNN mimarisi

Üçüncü model olarak `torchvision.models` içinden `ResNet18` seçilmiştir.

- Mimari: `ResNet18`
- Başlangıç ayarı: `pretrained=True`
- Son tam bağlı katman: `10` sınıfa göre yeniden yazılmıştır
- Eğitilen kısım: sadece `layer4` ve `fc`

Bu yapıda daha erken bloklar dondurulmuş, son blok ve sınıflandırıcı katmanı eğitilmiştir. Bu sayede hem hazır literatür mimarisi kullanılmış hem de eğitim maliyeti makul tutulmuştur.

### Model 5: Hibrit karşılaştırma için tam CNN

Beşinci model, hem doğrudan sınıflandırma yapan hem de özellik çıkarma görevi üstlenen tam CNN mimarisidir. Giriş yine `1 x 32 x 32` olarak tutulmuştur. Kullanılan yapı klasik CNN mantığı içindedir:

- `Conv2d(1, 32, 3, padding=1)`
- `BatchNorm2d`
- `ReLU`
- `Conv2d(32, 32, 3, padding=1)`
- `BatchNorm2d`
- `ReLU`
- `MaxPool2d(2)`
- `Conv2d(32, 64, 3, padding=1)`
- `BatchNorm2d`
- `ReLU`
- `Conv2d(64, 64, 3, padding=1)`
- `BatchNorm2d`
- `ReLU`
- `MaxPool2d(2)`
- `Conv2d(64, 128, 3, padding=1)`
- `BatchNorm2d`
- `ReLU`
- `Conv2d(128, 128, 3, padding=1)`
- `BatchNorm2d`
- `ReLU`
- `MaxPool2d(2)`
- `Flatten`
- `Linear(128 * 4 * 4, 256)`
- `ReLU`
- `Dropout`
- `Linear(256, 10)`

Bu modelin `extract_features()` çıktıları hibrit modelin girdi uzayını oluşturmuştur.

### Model 4: Hibrit model

Dördüncü model, beşinci modelden çıkarılan özelliklerle kurulmuştur. İş akışı şu şekilde ilerlemiştir:

1. Model 5 eğitildi
2. `extract_features()` ile eğitim ve test özellikleri alındı
3. Özellikler ve etiketler `.npy` dosyalarına yazıldı
4. Bu özelliklerle `LinearSVC` eğitildi ve test edildi

Oluşan dosyalar:

- `data/features/train_features.npy`
- `data/features/train_labels.npy`
- `data/features/test_features.npy`
- `data/features/test_labels.npy`

Oluşan boyutlar:

- Eğitim özellik matrisi: `(50000, 256)`
- Eğitim etiket vektörü: `(50000,)`
- Test özellik matrisi: `(10000, 256)`
- Test etiket vektörü: `(10000,)`

Böylece hibrit model maddesi, ödev metnindeki ".npy dosyalarına özellik ve label seti yazdırılacak, daha sonra kanonik bir makine öğrenmesi modeliyle eğitilecek" şartını doğrudan sağlamıştır.

### Kayıp fonksiyonu, optimizer ve hiperparametreler

Tüm CNN modellerinde kayıp fonksiyonu olarak `nn.CrossEntropyLoss()` kullanılmıştır. Optimizer olarak `Adam` seçilmiştir. Hiperparametreler, ödev metninin izin verdiği çerçevede ve doğrulama davranışına göre aşağıdaki gibi tutulmuştur:

| Model | Epoch | Learning Rate | Batch Size | Weight Decay | Giriş |
| --- | ---: | ---: | ---: | ---: | ---: |
| Model 1 | 20 | 0.0010 | 128 | 0.0 | 32 |
| Model 2 | 24 | 0.0008 | 128 | 0.0001 | 32 |
| Model 3 | 8 | 0.0001 | 64 | 0.0001 | 224 |
| Model 5 | 24 | 0.0010 | 128 | 0.0001 | 32 |

Bu seçimlerin temel nedeni şunlardır:

- `Adam`, bu veri setinde hızlı ve kararlı yakınama sağladı
- Model 2 için daha küçük öğrenme oranı, batch normalization ve dropout ile daha düzgün bir doğrulama davranışı verdi
- Model 3'te hazır ağ ağırlıkları kullanıldığı için öğrenme oranı daha düşük tutuldu
- Model 5'te daha güçlü temsil uzayı hedeflendiği için 24 epoch kullanıldı

### Tekrarlanabilirlik

Projeyi baştan çalıştırmak için:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pytest -q tests -p no:cacheprovider
python -m src.run_experiments
```

Ana çıktı dosyaları:

- `artifacts/reports/experiment_metrics.csv`
- `artifacts/reports/summary.json`
- `artifacts/reports/hybrid_feature_summary.json`
- `artifacts/plots/training_curves_cnn_models.png`
- `artifacts/plots/model_metric_comparison.png`
- tüm confusion matrix görselleri

## Sonuçlar

Deneyler CUDA destekli ortamda çalıştırılmıştır. Elde edilen test sonuçları aşağıdadır:

| Model | Best Epoch | Test Accuracy | Precision Macro | Recall Macro | F1 Macro | Test Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Model 1 - LeNet Baseline | 16 | 0.9901 | 0.9899 | 0.9900 | 0.9900 | 0.0336 |
| Model 2 - LeNet Improved | 22 | 0.9928 | 0.9928 | 0.9927 | 0.9927 | 0.0260 |
| Model 3 - ResNet18 Reference | 4 | 0.9937 | 0.9937 | 0.9936 | 0.9936 | 0.0208 |
| Model 4 - Hybrid LinearSVC | - | 0.9958 | 0.9958 | 0.9958 | 0.9958 | - |
| Model 5 - Full CNN | 24 | 0.9949 | 0.9948 | 0.9949 | 0.9948 | 0.0167 |

Bu sonuçlarda bütün modellerin `0.99` bandına çıktığı görülmektedir. En yüksek doğruluğu hibrit `LinearSVC` modeli vermiştir. Bunun anlamı, Model 5'in öğrendiği özellik uzayının doğrusal olarak da çok iyi ayrışıyor olmasıdır.

Split ve özellik raporları:

- Split: eğitim `50000`, doğrulama `10000`, test `10000`
- Normalizasyon mean/std: `[0.13101358806265007] / [0.2894179023904892]`
- `train_features.npy`: `(50000, 256)`
- `train_labels.npy`: `(50000,)`
- `test_features.npy`: `(10000, 256)`
- `test_labels.npy`: `(10000,)`

Üretilen görseller:

- `artifacts/plots/train_class_distribution.png`
- `artifacts/plots/training_curves_cnn_models.png`
- `artifacts/plots/model_metric_comparison.png`
- `artifacts/plots/confusion_matrix_model_1_lenet_baseline.png`
- `artifacts/plots/confusion_matrix_model_2_lenet_improved.png`
- `artifacts/plots/confusion_matrix_model_3_resnet18_reference.png`
- `artifacts/plots/confusion_matrix_model_4_hybrid_linear_svc.png`
- `artifacts/plots/confusion_matrix_model_5_full_cnn_for_hybrid_comparison.png`

## Tartışma

Sonuçlar birlikte değerlendirildiğinde birkaç net nokta ortaya çıkıyor.

İlk olarak, ders mantığına en yakın referans yapı olan Model 1 bile `0.9901` test doğruluğuna ulaşmıştır. Bu, `MNIST` veri setinin CNN tabanlı modeller için uygun ve temiz bir karşılaştırma zemini sunduğunu gösteriyor. Model 2 ise aynı çekirdek yapıyı koruyup sadece batch normalization ve dropout ekleyerek `0.9928` seviyesine çıkmıştır. Yani ödev metninde istenen iyileştirme mantığı gerçekten olumlu sonuç vermiştir.

Hazır mimari olan Model 3, `pretrained=True` ve sadece son bloklar eğitilecek şekilde kullanıldığında `0.9937` accuracy üretmiştir. Bu model, açık yazılan temel modellerden daha yüksek performans vermiştir; ancak en yüksek skor yine de hibrit ve tam CNN tarafında çıkmıştır. Bunun sebebi, proje veri setinin görece basit olması ve iyi öğrenilmiş özellik uzayının sınıfları çok net ayırabilmesidir.

En dikkat çekici sonuç Model 4 ve Model 5 karşılaştırmasında görülmüştür. Model 5 tek başına `0.9949` accuracy üretirken, aynı modelin çıkardığı özelliklerle eğitilen hibrit `LinearSVC` modeli `0.9958` accuracy vermiştir. Bu fark çok büyük değil ama anlamlı; çünkü ödev metnindeki hibrit model fikrinin sadece formalite olmadığını, gerçekten güçlü bir temsil çıkışına dönüşebildiğini gösteriyor.

Genel olarak bakıldığında ödev metnindeki tüm ana koşullar sağlanmıştır:

- Görüntü veri seti kullanıldı
- Ön işleme açıkça yapıldı
- İki açık CNN sınıfı yazıldı
- Hazır mimari kullanıldı
- Hibrit model `.npy` özellik dosyalarıyla kuruldu
- Aynı veri uzayında tam CNN ile karşılaştırma yapıldı
- Sonuçlar tablo ve görsellerle raporlandı

## Referanslar

1. LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. Gradient-Based Learning Applied to Document Recognition. *Proceedings of the IEEE*, 1998.
2. Deng, L. The MNIST Database of Handwritten Digit Images for Machine Learning Research. *IEEE Signal Processing Magazine*, 2012.
3. He, K., Zhang, X., Ren, S., Sun, J. Deep Residual Learning for Image Recognition. *CVPR*, 2016.
4. PyTorch Documentation, `torch.nn.Conv2d`
5. PyTorch Documentation, `torch.nn.CrossEntropyLoss`
6. Torchvision Documentation, `torchvision.models.resnet18`
