# IMDb Film Yorumlarında Duygu Analizi

Bu proje, Stanford IMDb Large Movie Review Dataset üzerinde film yorumlarını pozitif/negatif olarak sınıflandırır. Amaç yalnızca tek bir model kurmak değil; farklı önişleme ayarları ve farklı model mimarilerinin metrikler üzerindeki etkisini nesnel olarak karşılaştırmaktır.

## Proje Sorusu

Farklı önişleme teknikleri ve model mimarileri, IMDb film yorumu duygu analizinde başarıyı nasıl etkiler?

## Deney Tasarımı

Önişleme profilleri:

- `basic`: HTML temizleme + lowercase
- `no_lowercase`: HTML temizleme, büyük/küçük harf korunur
- `punctuation_removed`: HTML temizleme + lowercase + noktalama temizleme
- `stopwords_removed`: HTML temizleme + lowercase + noktalama temizleme + stopword temizleme

Modeller:

- `tfidf_lr`: TF-IDF + Logistic Regression
- `tfidf_nb`: TF-IDF + Multinomial Naive Bayes
- `tfidf_svm`: TF-IDF + Linear SVM
- `embedding_dense`: Embedding ortalaması + Dense sınıflandırıcı
- `textcnn`: 1D CNN for text
- `bilstm`: Bidirectional LSTM
- `cnn_bilstm`: CNN + BiLSTM hibrit model

Metrikler:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Eğitim süresi

## Kurulum

Önce sanal ortam oluştur:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Derin öğrenme modellerini de çalıştırmak için:

```bash
pip install -r requirements-neural.txt
```

Geliştirme/test bağımlılıkları için:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Hızlı Duman Testi

Küçük gömülü örnek veriyle kod akışını kontrol eder:

```bash
python -m src.run_experiments --sample --models tfidf_lr tfidf_nb --preprocess basic stopwords_removed
```

## Gerçek IMDb Verisiyle Çalıştırma

İlk çalıştırmada Stanford veri seti otomatik indirilir:

```bash
python -m src.run_experiments --download --models tfidf_lr tfidf_nb tfidf_svm --preprocess basic stopwords_removed
```

Daha hızlı bir gerçek veri denemesi için her sınıftan sınırlı örnek:

```bash
python -m src.run_experiments --download --limit-per-class 1000 --models tfidf_lr textcnn bilstm --preprocess basic punctuation_removed --epochs 3
```

Tüm sonuçlar `results/experiment_results.csv` dosyasına yazılır. Her deney için confusion matrix görselleri de `results/` altında üretilir.

## Sunumda Anlatılacak Ana Fikir

Bu proje, klasik ML ve derin öğrenme yaklaşımlarının aynı veri setinde, aynı metriklerle karşılaştırılmasını sağlar. TF-IDF tabanlı modeller güçlü ve hızlı baseline verirken; CNN yerel kelime kalıplarını, BiLSTM ise kelime sırası ve bağlam ilişkisini yakalamaya çalışır. Stopword temizleme gibi önişleme kararlarının duygu analizinde her zaman faydalı olmayabileceği özellikle tartışılabilir; çünkü `not good` gibi ifadelerde anlamı taşıyan küçük kelimeler kritik olabilir.

## Kaynaklar

- Stanford AI Lab, [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- Maas et al. (2011), [Learning Word Vectors for Sentiment Analysis](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)
- Kim (2014), [Convolutional Neural Networks for Sentence Classification](https://aclanthology.org/D14-1181/)
- Devlin et al. (2018), [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
