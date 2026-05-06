# IMDb Duygu Analizi Sunum Konusma Notlari

## Sure Plani

Bu sunum 5-7 dakika icin tasarlandi. Slayt 1-3 giris ve literatur, slayt 4-8 veri ve deney tasarimi, slayt 9-11 model/metrik teorisi, slayt 12-17 bulgular ve yorum, slayt 18 ise soru-cevapta acilacak ek tablo olarak kullanilabilir.

## Slayt 1 - Baslik

Bu projede IMDb film yorumlarinin pozitif mi negatif mi oldugunu tahmin eden modelleri karsilastirdik. Amac sadece bir model egitmek degil; farkli onisleme secimleri ve farkli mimarilerin ayni problemde nasil davrandigini nesnel metriklerle gostermek.

## Slayt 2 - Sunum Haritasi

Sunum akisi dort parcadan olusuyor: problem ve literatur, veri seti ve onisleme, model/metric karsilastirmasi, sonra da sonuc ve gelecek calisma. Bu akisi secmemizin nedeni, projenin deney agirlikli olmasi: once neden onemli, sonra nasil test ettik, en son ne bulduk.

## Slayt 3 - Problem ve Literatur

Duygu analizi, yorum gibi serbest metinleri sayisal ve karar verilebilir sinyale donusturur. Pang ve Lee, opinion mining ve sentiment analysis alaninin online yorumlar ve bloglar gibi fikir iceren kaynaklarin artmasiyla onem kazandigini vurgular. Maas ve arkadaslari ise IMDb Large Movie Review Dataset'i ikili sentiment benchmark'i olarak yayinlamistir. Kim'in TextCNN calismasi kisa metin siniflandirmada CNN filtrelerinin yerel ifade kaliplarini yakalayabildigini, BERT tarzi transformer modelleri ise baglama duyarlı temsilin daha ileri bir adim oldugunu gosterir.

## Slayt 4 - Dataset Yapisi

Kullandigimiz veri seti Stanford IMDb Large Movie Review Dataset. Resmi yapida 25.000 egitim ve 25.000 test yorumu var; siniflar dengeli, yani pozitif ve negatif sayilari esit. Bu denge accuracy'yi daha anlamli yapar ama yine de precision, recall ve F1'i raporladik cunku hata yonunu sadece accuracy ile goremeyiz.

## Slayt 5 - Kesifsel Veri Analizi

Yorum uzunluklari cok degisken: bazi yorumlar cok kisa, bazilari uzun paragraflardan olusuyor. Bu nedenle neural modellerde padding ve maksimum uzunluk secimi kritik hale geliyor. Cok kisa max_len bilgi kaybettirir, cok uzun max_len ise egitimi yavaslatir.

## Slayt 6 - Arastirma Sorulari

Uc temel soru sorduk: hangi model daha guclu, hangi onisleme secimi performansi degistiriyor, daha karmasik modeller ek maliyete degiyor mu? Bu sayede sunum tek bir sonuc tablosuna sikismiyor; model seciminin gerekcesini de tartisabiliyoruz.

## Slayt 7 - Deney Matrisi

Deneylerde 4 onisleme varyanti ve 8 model ailesi kullanildi. Onisleme tarafinda basic, punctuation_removed, stopwords_removed ve no_lowercase profilleri var. Model tarafinda majority baseline, TF-IDF + Logistic Regression, SVM, Naive Bayes, Embedding + Dense, TextCNN, BiLSTM ve CNN + BiLSTM var.

## Slayt 8 - Onisleme Hatti

Pipeline ham yorumdan basliyor: lowercase, noktalama temizleme, stopword temizleme ve padding gibi adimlar farkli senaryolarda deneniyor. Buradaki onemli bulgu su: stopword temizleme her zaman iyi degil. Cunku "not", "no", "never" gibi kelimeler sentiment icin tasiyici olabilir.

## Slayt 9 - Metin Temsili

Klasik modellerde TF-IDF kullandik; bu temsil bir kelimenin dokumanda ne kadar ayirt edici oldugunu agirliklandirir. Neural modellerde ise kelime indeksleri embedding katmanina girer; ardindan Dense, CNN veya LSTM yapilari bu temsilleri isler.

## Slayt 10 - Model Teorisi

TF-IDF + klasik ML hattinin avantaji hizli ve yorumlanabilir olmasi. TextCNN yerel kaliplari yakalar: "very boring", "not worth" gibi ifadeler buna ornek. BiLSTM iki yonde baglam okur, CNN + BiLSTM ise once yerel kalip sonra sirali baglam yakalamayi hedefler.

## Slayt 11 - Metrikler

Accuracy genel dogruluk oranidir. Precision, pozitif dediklerimizin ne kadarinin gercekten pozitif oldugunu; recall, gercek pozitiflerin ne kadarini yakaladigimizi gosterir. F1 ise precision ve recall arasinda dengeli bir ozet oldugu icin ana siralama metrigi olarak kullanildi.

## Slayt 12 - Model Siralamasi

En iyi sonuc TF-IDF + Logistic Regression modelinden geldi: yaklasik %87.87 F1. Bu, guclu bir klasik baseline'in ne kadar onemli oldugunu gosteriyor. Derin modeller bu kosuda daha dusuk kaldi; bunun nedeni sinirli epoch, sinirli alt orneklem ve henuz hiperparametre optimizasyonu yapilmamis olmasi.

## Slayt 13 - Onisleme Etkisi

Heatmap bize model ve onisleme etkisini ayni anda gosteriyor. Logistic Regression icin basic ve punctuation_removed neredeyse ayni zirvede. Stopword temizleme performansi biraz dusurdu. Bu, sentiment analizinde "fazla temizlik" yapmanin bazi anlam sinyallerini silebilecegini gosterir.

## Slayt 14 - Sure ve Performans

TF-IDF tabanli modeller cok daha hizli egitildi ve yuksek F1 verdi. Neural modeller daha fazla zaman istiyor, ama bu haliyle optimize edilmemis ilk deneme olarak okunmali. Gelecek calismada epoch sayisi, embedding boyutu, dropout, max_len ve tam veri seti ile tekrar denenebilir.

## Slayt 15 - Hata Analizi

En iyi model 2000 test orneginde 1757 dogru tahmin uretti. Confusion matrix'e bakinca negatif ve pozitif siniflarda hata sayilarinin birbirine yakin oldugunu goruyoruz. Soru-cevapta bu slayt uzerinden "model nerede yaniliyor?" sorusuna gecilebilir: ironi, uzun baglam ve olumsuzluk yapilari tipik hata kaynaklari olabilir.

## Slayt 16 - Sinirliliklar

Bu projenin sinirliliklari acik: deneyler hizli calisabilsin diye alt orneklemle yapildi, neural modeller sadece kisa egitim kosullarinda denendi, pretrained transformer dahil edilmedi. Bunlar projeyi zayiflatmak yerine gelecek calisma planini netlestiriyor.

## Slayt 17 - Sonuc ve Gelecek Calisma

Ana sonuc: Bu problemde sadece "en karmasik modeli" secmek dogru degil; once guclu ve adil bir karsilastirma kurmak gerekiyor. Mevcut kosulda TF-IDF + Logistic Regression en guclu baslangic cizgisini verdi. Gelecekte tam veri, hiperparametre aramasi, BERT/DistilBERT ve nitel hata analizi ile calisma genisletilebilir.

## Slayt 18 - Ek Tablo

Bu slayt ana anlatida gecilmek zorunda degil; soru gelirse acilabilir. Tum ana deneyleri, metrikleri ve egitim surelerini tek tabloda gosterir. Ozellikle "hangi model ikinci geldi?", "stopword temizleme ne kadar etkiledi?", "neural modeller neden dusuk?" gibi sorulara cevap verir.

## Soru-Cevap Icin Hazir Cevaplar

**Neden TF-IDF + Logistic Regression en iyi cikti?**  
Bu veri setinde kelime agirliklari sentiment icin cok guclu sinyal veriyor. Logistic Regression hizli, iyi regularize edilen ve sparse TF-IDF vektorleriyle uyumlu bir model. Neural modeller ise daha fazla veri, epoch ve tuning ister.

**Stopword temizleme neden performansi dusurebilir?**  
Sentiment analizinde bazi kisa kelimeler anlami tersine cevirir. "not good" ifadesinde "not" silinirse yorumun yonu bozulur. Bu nedenle stopword temizleme her NLP probleminde otomatik fayda saglamaz.

**Accuracy yerine neden F1?**  
Dataset dengeli oldugu icin accuracy tamamen anlamsiz degil, ama F1 precision ve recall dengesini gosterdigi icin model karsilastirmasinda daha guvenilir bir ozet verir.

**Neural modeller neden dusuk kaldi?**  
Bu kosuda neural modeller 2 epoch ve sinirli alt orneklemle calistirildi. Pretrained embedding veya transformer kullanilmadi. Bu nedenle son karar degil, ilk deney sonucu olarak yorumlanmali.

**Projeyi nasil iyilestirirdik?**  
Tam 25k train/25k test uzerinde calistirir, validation set ile hiperparametre arar, DistilBERT gibi subword tabanli bir modeli ekler, yanlis tahmin edilen ornekleri nitel olarak siniflandirirdik.

## Kaynaklar

- Pang, B. ve Lee, L. (2008). "Opinion Mining and Sentiment Analysis", Foundations and Trends in Information Retrieval. https://www.nowpublishers.com/article/Details/INR-011
- Maas, A. L. ve ark. (2011). "Learning Word Vectors for Sentiment Analysis", ACL. https://aclanthology.org/P11-1015/
- Stanford Large Movie Review Dataset. https://ai.stanford.edu/~amaas/data/sentiment/
- Kim, Y. (2014). "Convolutional Neural Networks for Sentence Classification".
- Devlin, J. ve ark. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding".
