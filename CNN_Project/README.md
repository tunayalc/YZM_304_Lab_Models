# YZM304 Derin Ogrenme Dersi Proje 2

**Ogrenci:** Ahmet Tunahan Yalcin  
**Ogrenci No:** 22290665  
**Bolum:** Ankara Universitesi Yapay Zeka ve Veri Muhendisligi  
**Donem:** 2025-2026 Bahar  

Bu depo, CNN kullanarak ozellik cikarma ve siniflandirma yapilan proje odevinin teslime hazir halidir. Calisma dogrudan odev metnindeki maddelere gore kurulmustur: iki acik yazilmis CNN sinifi, bir hazir literatur mimarisi, bir hibrit model ve ayni veri uzayinda karsilastirilan tam CNN modeli birlikte egitilip test edilmistir.

## Giris

Bu projede goruntu verisi uzerinde evrisimli sinir aglari ile siniflandirma ve ozellik cikarma calismasi yapilmistir. Veri seti olarak `MNIST` secilmistir. Bu secim odev metniyle dogrudan uyumludur; cunku metin benchmark veri setlerinden birinin kullanilabilecegini acikca soylemektedir. `MNIST`, tek kanalli ve dengeli bir veri seti oldugu icin hem LeNet-5 benzeri acik CNN yapilarini hem de hazir CNN mimarilerini temiz bicimde karsilastirmaya uygun bir taban sunmaktadir.

Bu calismada bes model birlikte ele alinmistir:

1. LeNet-5 benzeri temel CNN
2. Ayni cekirdek katman hiperparametrelerini koruyan, batch normalization ve dropout eklenmis iyilestirilmis CNN
3. Hazir mimari olarak kullanilan `ResNet18`
4. Tam CNN'den cikarilan ozelliklerle egitilen hibrit `LinearSVC`
5. Hibrit model ile ayni veri uzayinda karsilastirilan tam CNN modeli

Boylece hem ders kapsamindaki temel CNN mantigi hem de daha guclu temsil uzayi ureten hazir ve hibrit yaklasimlar ayni problem uzerinde birlikte degerlendirilmis oldu.

## Yontem

### Veri seti ve bolme

- Veri seti: `torchvision.datasets.MNIST`
- Goruntu tipi: gri seviye
- Standart giris boyutu: `1 x 32 x 32`
- Sinif sayisi: `10`
- Siniflar: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`

Veri bolmesi `seed = 42` ile ve `StratifiedShuffleSplit` kullanilarak yapilmistir:

- Egitim: `50.000`
- Dogrulama: `10.000`
- Test: `10.000`

Bolme bilgileri `data/splits/split_manifest.json` dosyasina yazdirilmistir.

### On isleme

Uygulanan on isleme adimlari sadece odev metninin izin verdigi cekirdek akista tutulmustur:

1. Goruntulerin tensore cevrilmesi
2. Egitim alt kumesinden mean/std hesaplanmasi
3. Normalizasyon
4. Boyutun `32 x 32` olacak sekilde eslenmesi

Egitim alt kumesinden hesaplanan normalizasyon degerleri:

- Mean: `[0.13101358806265007]`
- Std: `[0.2894179023904892]`

Model 1, Model 2 ve Model 5 icin giris `1 x 32 x 32` tutulmustur.  
Model 3 icin tek kanalli goruntu 3 kanala kopyalanmis ve `224 x 224` boyutuna getirilmistir. Bu tercih, hazir `ResNet18` mimarisini odev metninden sapmadan kullanabilmek icin yapilmistir.

Odev metninde acikca yer almayan ek veri cogaltma adimlari kullanilmamistir.

### Model 1: LeNet benzeri temel CNN

Ilk model acik sinif olarak yazilmistir. Kullanilan katman yapisi:

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

Bu model dogrudan temel referans model olarak kullanilmistir.

### Model 2: Iyilestirilmis CNN

Ikinci modelde ilk modelin cekirdek konvolusyon yapisi korunmustur. Aglari iyilestirmesi beklenen ozel katmanlar uygun yerlere eklenmistir:

- `BatchNorm2d`
- `BatchNorm1d`
- `Dropout`

Bu nedenle odev metnindeki "ilk modeldeki katmanlarin hiperparametreleri ayni kalacak, sadece batch normalization ya da dropout gibi ozel katmanlar eklenebilir" kosulu dogrudan saglanmistir.

### Model 3: Hazir CNN mimarisi

Ucuncu model olarak `torchvision.models` icinden `ResNet18` secilmistir.

- Mimari: `ResNet18`
- Baslangic ayari: `pretrained=True`
- Son tam bagli katman: `10` sinifa gore yeniden yazilmistir
- Egitilen kisim: sadece `layer4` ve `fc`

Bu yapida daha erken bloklar dondurulmus, son blok ve siniflandirici katmani egitilmistir. Bu sayede hem hazir literatur mimarisi kullanilmis hem de egitim maliyeti makul tutulmustur.

### Model 5: Hibrit karsilastirma icin tam CNN

Besinci model, hem dogrudan siniflandirma yapan hem de ozellik cikarma gorevi ustlenen tam CNN mimarisidir. Giris yine `1 x 32 x 32` olarak tutulmustur. Kullanilan yapi klasik CNN mantigi icindedir:

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

Bu modelin `extract_features()` ciktilari hibrit modelin girdi uzayini olusturmustur.

### Model 4: Hibrit model

Dorduncu model, besinci modelden cikarilan ozelliklerle kurulmustur. Is akis su sekilde ilerlemistir:

1. Model 5 egitildi
2. `extract_features()` ile egitim ve test ozellikleri alindi
3. Ozellikler ve etiketler `.npy` dosyalarina yazildi
4. Bu ozelliklerle `LinearSVC` egitildi ve test edildi

Olusan dosyalar:

- `data/features/train_features.npy`
- `data/features/train_labels.npy`
- `data/features/test_features.npy`
- `data/features/test_labels.npy`

Olusan boyutlar:

- Egitim ozellik matrisi: `(50000, 256)`
- Egitim etiket vektoru: `(50000,)`
- Test ozellik matrisi: `(10000, 256)`
- Test etiket vektoru: `(10000,)`

Boylece hibrit model maddesi, odev metnindeki ".npy dosyalarina ozellik ve label seti yazdirilacak, daha sonra kanonik bir makine ogrenmesi modeliyle egitilecek" sartini dogrudan saglamistir.

### Kayip fonksiyonu, optimizer ve hiperparametreler

Tum CNN modellerinde kayip fonksiyonu olarak `nn.CrossEntropyLoss()` kullanilmistir. Optimizer olarak `Adam` secilmistir. Hiperparametreler, odev metninin izin verdigi cercevede ve dogrulama davranisina gore asagidaki gibi tutulmustur:

| Model | Epoch | Learning Rate | Batch Size | Weight Decay | Giris |
| --- | ---: | ---: | ---: | ---: | ---: |
| Model 1 | 20 | 0.0010 | 128 | 0.0 | 32 |
| Model 2 | 24 | 0.0008 | 128 | 0.0001 | 32 |
| Model 3 | 8 | 0.0001 | 64 | 0.0001 | 224 |
| Model 5 | 24 | 0.0010 | 128 | 0.0001 | 32 |

Bu secimlerin temel nedeni sunlardir:

- `Adam`, bu veri setinde hizli ve kararlı yakinama sagladi
- Model 2 icin daha kucuk ogrenme orani, batch normalization ve dropout ile daha duzgun bir dogrulama davranisi verdi
- Model 3'te hazir ag agirliklari kullanildigi icin ogrenme orani daha dusuk tutuldu
- Model 5'te daha guclu temsil uzayi hedeflendigi icin 24 epoch kullanildi

### Tekrarlanabilirlik

Projeyi bastan calistirmak icin:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pytest -q tests -p no:cacheprovider
python -m src.run_experiments
```

Ana cikti dosyalari:

- `artifacts/reports/experiment_metrics.csv`
- `artifacts/reports/summary.json`
- `artifacts/reports/hybrid_feature_summary.json`
- `artifacts/plots/training_curves_cnn_models.png`
- `artifacts/plots/model_metric_comparison.png`
- tum confusion matrix gorselleri

## Sonuclar

Deneyler CUDA destekli ortamda calistirilmistir. Elde edilen test sonuclari asagidadir:

| Model | Best Epoch | Test Accuracy | Precision Macro | Recall Macro | F1 Macro | Test Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Model 1 - LeNet Baseline | 16 | 0.9901 | 0.9899 | 0.9900 | 0.9900 | 0.0336 |
| Model 2 - LeNet Improved | 22 | 0.9928 | 0.9928 | 0.9927 | 0.9927 | 0.0260 |
| Model 3 - ResNet18 Reference | 4 | 0.9937 | 0.9937 | 0.9936 | 0.9936 | 0.0208 |
| Model 4 - Hybrid LinearSVC | - | 0.9958 | 0.9958 | 0.9958 | 0.9958 | - |
| Model 5 - Full CNN | 24 | 0.9949 | 0.9948 | 0.9949 | 0.9948 | 0.0167 |

Bu sonuclarda butun modellerin `0.99` bandina ciktigi gorulmektedir. En yuksek dogrulugu hibrit `LinearSVC` modeli vermistir. Bunun anlami, Model 5'in ogrendigi ozellik uzayinin dogrusal olarak da cok iyi ayrisiyor olmasidir.

Split ve ozellik raporlari:

- Split: egitim `50000`, dogrulama `10000`, test `10000`
- Normalizasyon mean/std: `[0.13101358806265007] / [0.2894179023904892]`
- `train_features.npy`: `(50000, 256)`
- `train_labels.npy`: `(50000,)`
- `test_features.npy`: `(10000, 256)`
- `test_labels.npy`: `(10000,)`

Uretilen gorseller:

- `artifacts/plots/train_class_distribution.png`
- `artifacts/plots/training_curves_cnn_models.png`
- `artifacts/plots/model_metric_comparison.png`
- `artifacts/plots/confusion_matrix_model_1_lenet_baseline.png`
- `artifacts/plots/confusion_matrix_model_2_lenet_improved.png`
- `artifacts/plots/confusion_matrix_model_3_resnet18_reference.png`
- `artifacts/plots/confusion_matrix_model_4_hybrid_linear_svc.png`
- `artifacts/plots/confusion_matrix_model_5_full_cnn_for_hybrid_comparison.png`

## Tartisma

Sonuclar birlikte degerlendirildiginde birkac net nokta ortaya cikiyor.

Ilk olarak, ders mantigina en yakin referans yapi olan Model 1 bile `0.9901` test dogruluguna ulasmistir. Bu, `MNIST` veri setinin CNN tabanli modeller icin uygun ve temiz bir karsilastirma zemini sundugunu gosteriyor. Model 2 ise ayni cekirdek yapiyi koruyup sadece batch normalization ve dropout ekleyerek `0.9928` seviyesine cikmistir. Yani odev metninde istenen iyilestirme mantigi gercekten olumlu sonuc vermistir.

Hazir mimari olan Model 3, `pretrained=True` ve sadece son bloklar egitilecek sekilde kullanildiginda `0.9937` accuracy uretmistir. Bu model, acik yazilan temel modellerden daha yuksek performans vermistir; ancak en yuksek skor yine de hibrit ve tam CNN tarafinda cikmistir. Bunun sebebi, proje veri setinin gorece basit olmasi ve iyi ogrenilmis ozellik uzayinin siniflari cok net ayirabilmesidir.

En dikkat cekici sonuc Model 4 ve Model 5 karsilastirmasinda gorulmustur. Model 5 tek basina `0.9949` accuracy uretirken, ayni modelin cikardigi ozelliklerle egitilen hibrit `LinearSVC` modeli `0.9958` accuracy vermistir. Bu fark cok buyuk degil ama anlamli; cunku odev metnindeki hibrit model fikrinin sadece formalite olmadigini, gercekten guclu bir temsil cikisina donusebildigini gosteriyor.

Genel olarak bakildiginda odev metnindeki tum ana kosullar saglanmistir:

- Goruntu veri seti kullanildi
- On isleme acikca yapildi
- Iki acik CNN sinifi yazildi
- Hazir mimari kullanildi
- Hibrit model `.npy` ozellik dosyalariyla kuruldu
- Ayni veri uzayinda tam CNN ile karsilastirma yapildi
- Sonuclar tablo ve gorsellerle raporlandi

## Referanslar

1. LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. Gradient-Based Learning Applied to Document Recognition. *Proceedings of the IEEE*, 1998.
2. Deng, L. The MNIST Database of Handwritten Digit Images for Machine Learning Research. *IEEE Signal Processing Magazine*, 2012.
3. He, K., Zhang, X., Ren, S., Sun, J. Deep Residual Learning for Image Recognition. *CVPR*, 2016.
4. PyTorch Documentation, `torch.nn.Conv2d`
5. PyTorch Documentation, `torch.nn.CrossEntropyLoss`
6. Torchvision Documentation, `torchvision.models.resnet18`
