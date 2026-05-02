# 🍛 Klasifikasi Gambar Makanan Tradisional Indonesia

[![Open In Colab - Fine Tuning](https://colab.research.google.com/assets/colab-badge.svg)](#)

## Latar Belakang
Proyek ini merupakan solusi *machine learning* yang dikembangkan untuk kompetisi **ACTION 2025**, sebuah ajang yang diselenggarakan oleh **HIMA Sains Data Universitas Negeri Surabaya (UNESA)**. 

Target utama dari kompetisi ini adalah membangun model *Computer Vision* yang mampu mengenali dan mengklasifikasikan berbagai gambar makanan tradisional khas Indonesia dengan tingkat akurasi yang tinggi, guna melestarikan dan mendigitalisasi pengenalan ragam kuliner nusantara.

---

## Permasalahan Teknis
Tantangan utama dalam kompetisi ini adalah kondisi dataset yang diberikan berupa kumpulan gambar yang **tidak memiliki label (*unlabeled*)**. 

**Distribusi Dataset:**
*   **Data Train:** 4.257 gambar
*   **Data Test:** 2.057 gambar
*   **Total Kelas:** 15 Kategori Makanan

**Pemetaan Kelas (ID to Label):**
```python 
ID2LABEL = {
    0: 'Ayam Bakar',     
    1: 'Ayam Betutu',
    2: 'Ayam Goreng',    
    3: 'Ayam Pop',
    4: 'Bakso',          
    5: 'Coto Makassar',
    6: 'Gado Gado',      
    7: 'Gudeg',
    8: 'Nasi Goreng',    
    9: 'Pempek',
    10: 'Rawon',         
    11: 'Rendang',
    12: 'Sate Madura',   
    13: 'Sate Padang',
    14: 'Soto'
}
```

---

## Metode
Untuk mengatasi masalah ketiadaan label pada ribuan data gambar, kami merancang *pipeline* pelatihan dua tahap yang cermat. Arsitektur model utama yang digunakan dalam proyek ini adalah **`CoAtNet-2-rw-224`**, yang menggabungkan keunggulan *Convolutional Neural Networks* (CNN) dan *Transformers* untuk pengenalan pola visual yang optimal.

Keseluruhan proses dibagi menjadi dua *notebook* utama:

1.  **Tahap 1: Semi-Supervised Learning (Notebook 1)**
    
    [![Open In Colab - Semi-Supervised](https://colab.research.google.com/assets/colab-badge.svg)](#)

    Karena data latih tidak berlabel, kami menggunakan pendekatan *Semi-Supervised Learning* untuk menghasilkan *pseudo-labels*. Proses ini dilakukan secara ketat melalui **7 kali iterasi** secara bertahap untuk memastikan model dapat belajar dan melabeli data mentah dengan tingkat kepercayaan (*confidence*) yang semakin tinggi di setiap iterasinya tanpa *overfitting* pada prediksinya sendiri.

2.  **Tahap 2: Full Parameter Fine-Tuning (Notebook 2)**

    [![Open In Colab - Fine Tuning](https://colab.research.google.com/assets/colab-badge.svg)](#)

    Setelah dataset memiliki *pseudo-labels* yang stabil dari tahap pertama, kami melakukan *Full Parameter Fine-Tuning*. Pada tahap ini, seluruh bobot (*weights*) dari model CoAtNet dilatih ulang secara menyeluruh menggunakan dataset yang telah dilabeli untuk mengoptimalkan representasi visual spesifik dari makanan tradisional. Proses evaluasi model juga dilakukan pada tahapan ini.

Download Model: [![Download Model](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)](https://huggingface.co/rakan416/CoAtNet-2-traditional-food-classification)
---

## Hasil Evaluasi
Evaluasi model dilakukan secara internal dengan memisahkan sebagian kecil data latih (yang telah memiliki label yang diyakini benar) sebagai himpunan validasi (*validation split*) sebanyak 401 gambar. 

Model berhasil mencapai **Akurasi Global 91%**, dengan performa yang sangat seimbang di berbagai metrik klasifikasi.

**Classification Report:**
```text
               precision    recall  f1-score   support

   Ayam Bakar       0.88      0.74      0.81        31
  Ayam Betutu       0.92      0.82      0.87        28
  Ayam Goreng       0.74      0.87      0.80        30
     Ayam Pop       1.00      0.96      0.98        23
        Bakso       1.00      1.00      1.00        22
Coto Makassar       0.91      0.94      0.92        31
    Gado Gado       0.94      0.94      0.94        32
        Gudeg       0.88      0.88      0.88        26
  Nasi Goreng       0.97      0.93      0.95        30
       Pempek       1.00      1.00      1.00        22
        Rawon       0.88      0.84      0.86        25
      Rendang       0.89      0.96      0.92        25
  Sate Madura       0.91      0.91      0.91        33
  Sate Padang       0.90      0.93      0.92        30
         Soto       0.80      0.92      0.86        13

     accuracy                           0.91       401
    macro avg       0.91      0.91      0.91       401
 weighted avg       0.91      0.91      0.91       401
```

---

## Kutipan
> *"If we want machines to think, we need to teach them to see."* 
> — **Fei-Fei Li** (Pionir Computer Vision & Pencipta ImageNet)

---

## Pengembang
Proyek ini dikembangkan secara kolaboratif oleh:
*   **Rakan R. Dewangga**
*   **Moch. Faiz Febriawan**
*   **M. Raihan Firdaus**