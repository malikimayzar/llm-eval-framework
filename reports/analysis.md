
---

## Finding 6: Jawaban Terlalu Singkat untuk Dievaluasi

### Observasi
Dari run pertama (mistral, clean dataset), dua case menghasilkan jawaban yang benar secara faktual tapi tidak bisa dievaluasi:

- **clean_008**: `'/files/home/johndoe/myfile.txt'` — path string, bukan kalimat deklaratif
- **clean_009**: `'PUT'` — satu kata, tidak ada struktur klaim

Kedua jawaban ini menghasilkan `score=0.0` bukan karena salah, tapi karena claim extractor tidak bisa mengekstrak klaim yang evaluable.

### Fix
`MIN_CLAIM_LENGTH_WORDS` diturunkan dari 4 ke 2. Tapi ini hanya partial fix — jawaban single-word seperti `'PUT'` tetap tidak bisa dievaluasi faithfulness-nya karena tidak ada klaim yang bisa di-trace ke konteks.

### Implikasi
Faithfulness evaluation memiliki blind spot untuk jawaban yang benar tapi ekstremal singkat. Untuk kasus ini, evaluasi yang tepat adalah exact match terhadap `ground_truth`, bukan semantic similarity. Ini area pengembangan selanjutnya.

---

## Finding 7: Timeout Tidak Konsisten pada CPU Inference

### Observasi
Dari 10 cases, 6 timeout setelah 600s. Tapi beberapa case yang sama berhasil diselesaikan Mistral dalam 380-648s. Artinya bukan konteks yang terlalu panjang — ada variabilitas yang tidak terprediksi dalam CPU inference time.

### Kemungkinan Penyebab
- Memory pressure saat model pertama kali di-load ke RAM
- Thermal throttling setelah beberapa query berturut-turut
- Panjang output yang bervariasi — query yang memicu jawaban panjang butuh waktu lebih lama

### Fix
Timeout dinaikkan ke 900s untuk run kedua. Jika masih ada timeout, evaluasi perlu dijalankan dengan jeda antar query untuk memberi CPU waktu recovery.
