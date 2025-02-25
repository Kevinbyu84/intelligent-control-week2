# intelligent-control-week2
Proyek ini berisi implementasi dua model Machine Learning (SVM dan KNN) untuk mendeteksi warna dari input kamera secara real-time menggunakan dataset warna dalam format CSV.

## Deskripsi Proyek
1. **ML_SVM.py**:
   - Membangun dan melatih model SVM untuk klasifikasi warna.
   - Mendeteksi dua area warna secara real-time dari input kamera.
   - Menampilkan prediksi warna dan akurasi secara langsung di layar.

2. **ML_KNN.py**:
   - Membangun dan melatih model KNN untuk klasifikasi warna.
   - Mendeteksi warna dari titik tengah frame kamera secara real-time.

## Dataset
Dataset yang digunakan adalah file CSV (`colors.csv`) yang berisi kolom berikut:

```
B,G,R,color_name
```
Contoh isi file `colors.csv`:
```
255,0,0,Red
0,255,0,Green
0,0,255,Blue
```

## Kustomisasi
- Anda dapat menyesuaikan ukuran bounding box atau metode prediksi dengan memodifikasi bagian kode di masing-masing skrip.
- Tambahkan lebih banyak warna ke `colors.csv` untuk meningkatkan akurasi model.