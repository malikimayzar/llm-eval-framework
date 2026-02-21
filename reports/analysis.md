
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

---

## Hasil Eval Lengkap — Mistral, Clean Dataset (Run 2)

**Run ID:** 20260221_074512  
**Total waktu:** 160 menit  
**Timeout setting:** 900s

| Case | Score | Keterangan |
|------|-------|-----------|
| clean_001 | 0.0 | FALSE INSUFFICIENT_CONTEXT (sim=0.7249) |
| clean_002 | 1.0 | Perfect |
| clean_003 | 1.0 | Perfect |
| clean_004 | 0.0 | FALSE INSUFFICIENT_CONTEXT (sim=0.9691) — kasus ekstrem |
| clean_005 | 1.0 | Perfect |
| clean_006 | 0.667 | Hallucination terdeteksi: tambah info OpenAPI schema |
| clean_007 | 1.0 | Perfect |
| clean_008 | 0.0 | Jawaban terlalu singkat: '/files/home/johndoe/myfile.txt' |
| clean_009 | 0.0 | Jawaban terlalu singkat: 'PUT' |
| clean_010 | 1.0 | Perfect |

**Avg faithfulness score: 0.567 (56.7%)**

### Observasi Kritis

**clean_004** adalah failure case paling ekstrem: similarity ground_truth ke konteks 0.9691 — hampir sempurna — tapi model menjawab `INSUFFICIENT_CONTEXT` tanpa mencoba. Ini bukan ambiguitas, ini kegagalan total membaca konteks.

**clean_006** adalah satu-satunya genuine hallucination yang tertangkap: model menambahkan *"Documentation of the max_length parameter in the OpenAPI schema"* yang tidak ada di konteks. Framework berhasil mendeteksi ini dengan score 0.667.

**clean_008 dan clean_009** mengekspos blind spot evaluator: jawaban factually correct tapi tidak evaluable karena terlalu singkat. Exact match terhadap ground_truth adalah solusi yang tepat untuk kasus ini.

---

## Hasil Eval Lengkap — Mistral, Distractor Dataset

**Run ID:** 20260221_103528
**Total waktu:** 120 menit

| Case | Score | Jawaban Model | Keterangan |
|------|-------|---------------|-----------|
| distractor_001 | 0.0 | skip=5, limit=100 | Mengikuti nilai yang dimanipulasi |
| distractor_002 | 0.0 | POST | Mengikuti HTTP method yang dibalik |
| distractor_003 | N/A | (kosong) | Inference failed |
| distractor_004 | 0.0 | :filepath | Jawaban terlalu singkat |
| distractor_005 | 0.333 | 3 poin, 1 hallucination | Tambah "caching" yang tidak ada di konteks |

**Avg faithfulness: 8.3% — turun dari 56.7% di clean dataset**

### Temuan Utama

**Mistral faithful ke konteks, termasuk konteks yang salah.**

distractor_001 dan distractor_002 membuktikan ini: model tidak menggunakan training knowledge untuk "koreksi" konteks yang dimanipulasi. Ketika konteks bilang `skip=5, limit=100`, model menjawab persis itu — meski nilai asli yang benar adalah `skip=0, limit=10`.

Ini adalah behavior yang diharapkan dari RAG system yang well-behaved. Tapi juga berarti: **garbage in, garbage out**. Kualitas retrieval menentukan kualitas jawaban.

### Perbandingan Clean vs Distractor

| Metrik | Clean Dataset | Distractor Dataset |
|--------|--------------|-------------------|
| Avg faithfulness | 56.7% | 8.3% |
| Perfect score | 5/10 | 0/5 |
| Failure rate | 50% | 100% |
| False INSUFFICIENT_CTX | 2 | 0 |

Penurunan 56.7% → 8.3% mengkonfirmasi bahwa evaluator sensitif terhadap manipulasi konteks — bukan hanya mengukur topical similarity.
