# Sunum Taslağı: IMDb Film Yorumlarında Duygu Analizi

Hedef süre: 5-7 dakika sunum + 2-3 dakika soru cevap.

## 1. Başlık ve Problem

IMDb film yorumlarında duygu analizi: Bir yorumu pozitif ya da negatif olarak otomatik sınıflandırmak.

Ana mesaj: İnternetteki yorum sayısı çok yüksek olduğu için metinlerden otomatik duygu çıkarımı; öneri sistemleri, ürün/film değerlendirme ve kullanıcı memnuniyeti analizi için önemlidir.

## 2. Literatür ve Gerekçe

- Sentiment analysis, NLP içinde metinden görüş/duygu çıkarma problemidir.
- Maas et al. (2011), IMDb Large Movie Review Dataset'i ikili sentiment classification için benchmark olarak sunmuştur.
- Kim (2014), CNN mimarilerinin cümle sınıflandırmada güçlü ve hızlı sonuçlar verebildiğini göstermiştir.
- Transformer tabanlı modeller, özellikle BERT, bağlama duyarlı temsil öğrenimiyle modern NLP'de güçlü bir gelecek çalışma yönüdür.

## 3. Dataset

- Kaynak: Stanford Large Movie Review Dataset.
- Eğitim: 25.000 yorum.
- Test: 25.000 yorum.
- Etiketler: pozitif ve negatif.
- Veri dengeli olduğu için accuracy anlamlıdır; yine de precision, recall ve F1 birlikte raporlanır.

## 4. Önişleme Deneyleri

Karşılaştırılan profiller:

- Basic: HTML temizleme + lowercase.
- Noktalama temizleme.
- Stopword temizleme.
- Lowercase yapılmayan versiyon.

Beklenen tartışma: Stopword temizleme duygu analizinde her zaman iyi değildir; çünkü "not good" gibi yapılarda küçük kelimeler duygu yönünü değiştirebilir.

## 5. Modeller

- Majority baseline: Veri dengesine göre en sık sınıfı tahmin eder.
- TF-IDF + Logistic Regression: Hızlı ve güçlü klasik ML baseline.
- TF-IDF + Naive Bayes / Linear SVM: Farklı klasik sınıflandırıcıların karşılaştırılması.
- 1D CNN: Metindeki yerel n-gram benzeri kalıpları yakalar.
- BiLSTM: Kelime sırası ve bağlam bilgisini iki yönden işler.
- CNN + BiLSTM: Önce yerel kalıplar, sonra sıralı bağlam.

## 6. Sonuçlar

Tablo önerisi:

| Model | Önişleme | Accuracy | Precision | Recall | F1 | Süre |
|---|---|---:|---:|---:|---:|---:|
| majority | basic | ... | ... | ... | ... | ... |
| tfidf_lr | basic | ... | ... | ... | ... | ... |
| textcnn | basic | ... | ... | ... | ... | ... |
| bilstm | basic | ... | ... | ... | ... | ... |

Ek görsel:

- En iyi modelin confusion matrix'i.
- Review uzunluğu dağılımı.

## 7. Yorum ve Gelecek Çalışmalar

- TF-IDF tabanlı klasik modeller hızlı ve güçlü baseline sağlayabilir.
- Derin modeller daha fazla veri ve eğitim süresiyle bağlamı daha iyi yakalayabilir.
- Stopword temizleme sonucu düşürebilir; özellikle negation kelimeleri korunmalıdır.
- Gelecekte BERT/DistilBERT, subword tokenization, hiperparametre optimizasyonu ve hata analizi yapılabilir.

## Kaynaklar

- Maas, A. L. et al. (2011). Learning Word Vectors for Sentiment Analysis.
- Stanford AI Lab. Large Movie Review Dataset.
- Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification.
- Devlin, J. et al. (2018). BERT.
