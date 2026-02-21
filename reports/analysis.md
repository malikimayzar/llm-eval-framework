# Evaluation Analysis Report
## LLM Faithfulness & Consistency Framework — v1.0

**Model evaluated:** phi3:mini (development), mistral (eval target)  
**Dataset:** FastAPI documentation (20 clean cases, 5 distractor cases, 6 paraphrase cases)  
**Infrastructure:** CPU-only, 16GB RAM, Ollama local inference  
**Date:** February 2026

---

## Overview

Framework ini dirancang untuk menjawab satu pertanyaan yang jarang ditanyakan secara serius:

> *Apakah jawaban LLM benar-benar berasal dari dokumen yang diberikan, atau hanya terlihat benar karena model sudah melihat data serupa saat training?*

Laporan ini mendokumentasikan observasi empiris dari menjalankan framework terhadap data nyata — termasuk failure case yang tidak disembunyikan, limitation yang ditemukan selama development, dan implikasi dari setiap temuan.

---

## Finding 1: False INSUFFICIENT_CONTEXT

### Observasi
phi3:mini secara konsisten menjawab `INSUFFICIENT_CONTEXT` untuk `clean_001` — pertanyaan tentang nilai default `skip` dan `limit` dalam URL contoh — meskipun konteks secara eksplisit menyebutkan:

> *"the query parameters are: skip with a value of 0, and limit with a value of 10"*

Jawaban model:
> *"The context does not provide information about specific default values for the `skip` and `limit` query parameters in this example URL, so INSUFFICIENT_CONTEXT."*

### Analisis
Ini bukan model yang jujur mengakui keterbatasan. Ini model yang **gagal membaca konteks dengan benar**, lalu berlindung di balik escape hatch yang disediakan prompt.

Framework versi pertama memberi score 1.0 untuk behavior ini — karena mengakui keterbatasan dianggap faithful. Setelah menemukan false positive ini, framework diperbarui dengan **validasi dua tahap**:

1. Jika model bilang `INSUFFICIENT_CONTEXT`, cek similarity antara `ground_truth` dan konteks menggunakan embedding.
2. Jika similarity ≥ 0.65, evidence ada di konteks — model gagal membaca. Ini `FALSE_INSUFFICIENT_CONTEXT` dengan score 0.0.

### Implikasi
Prompt yang terlalu agresif memberikan escape hatch (`INSUFFICIENT_CONTEXT`) mendorong model untuk under-answer. Ini tradeoff yang perlu dipertimbangkan dalam desain prompt RAG system.

### Threshold yang Ditemukan
- Evidence matching threshold: **0.75** (untuk klaim panjang)  
- Insufficient context validation threshold: **0.65** (lebih rendah karena ground truth biasanya lebih ringkas dari span konteks)

Kedua threshold ini berbeda secara fundamental dan tidak bisa disamakan.

---

## Finding 2: Claim Extraction Gagal pada Enumeration List

### Observasi
Jawaban model:
> *"The OpenAPI schema includes your API paths, parameters, authentication methods, response schemas, and rate limiting configuration."*

Claim extractor versi pertama mengekstrak ini sebagai **satu klaim**, bukan lima klaim atomik. Akibatnya, evaluasi faithfulness tidak granular — satu klaim yang berisi lima item dievaluasi sekaligus, dan item yang tidak didukung konteks ikut "lolos" bersama item yang valid.

### Fix
Ditambahkan `_expand_enumeration()` — fungsi yang mendeteksi pola `"X includes A, B, C, and D"` dan mengekspansinya menjadi klaim atomik:

- `"The OpenAPI schema includes API paths."`
- `"The OpenAPI schema includes parameters."`  
- `"The OpenAPI schema includes authentication methods."` ← tidak didukung konteks
- `"The OpenAPI schema includes response schemas."` ← tidak didukung konteks
- `"The OpenAPI schema includes rate limiting configuration."` ← tidak didukung konteks

Setelah fix, faithfulness score berubah dari **1.0 → 0.6** untuk jawaban yang sama. Dua klaim yang tidak didukung konteks kini terdeteksi.

### Implikasi
Granularitas claim extraction sangat menentukan kualitas evaluasi. Evaluator yang mengevaluasi kalimat panjang sebagai satu unit akan banyak menghasilkan false positive.

---

## Finding 3: nomic-embed-text Menganggap Paraphrase sebagai Semantically Berbeda

### Observasi
Dalam consistency evaluator, dua jawaban berikut dianggap semantically berbeda (similarity: **0.7677**, di bawah threshold 0.80):

- Jawaban A: `"By not declaring any default value for the parameter."`
- Jawaban B: `"You can make it required by not providing a default value."`

Keduanya memiliki makna yang identik.

### Analisis
Ini bukan bug framework — ini **limitation dari embedding model**. nomic-embed-text dioptimalkan untuk document-level similarity, bukan sentence-level semantic equivalence. Kalimat pendek dengan kata kerja berbeda (`declaring` vs `providing`, `default value` vs `default value`) menghasilkan embedding yang cukup berbeda.

**Implikasinya untuk framework:**
- Threshold 0.80 terlalu ketat untuk paraphrase detection di level kalimat
- Threshold diturunkan ke **0.72** setelah observasi empiris
- Masih ada false positive tersisa: "by not providing a default value" vs "simply omit the default value declaration" (similarity: 0.7756)

### Implikasi
Semantic similarity berbasis embedding tidak cukup untuk mendeteksi semantic equivalence pada kalimat pendek dan padat. Untuk domain teknis yang menggunakan terminologi spesifik, lexical overlap (ROUGE-L) memberikan sinyal komplementer yang penting.

---

## Finding 4: Semantic Similarity Tidak Mendeteksi Factual Inconsistency

### Observasi
Dalam retrieval dependency evaluator, jawaban `"PUT is used to update data"` vs `"PATCH is the correct HTTP method for partial updates"` menunjukkan similarity **0.8337** — di atas threshold, dianggap "konsisten".

Padahal keduanya memberikan jawaban yang berbeda untuk pertanyaan yang sama.

### Analisis
Embedding model mendeteksi bahwa kedua kalimat membahas topik yang sama (HTTP methods untuk update data). Tapi tidak bisa mendeteksi bahwa `PUT ≠ PATCH` secara faktual.

Ini adalah **fundamental limitation dari pendekatan embedding-based evaluation**: similarity topik ≠ similarity fakta.

### Implikasi
Untuk mendeteksi factual inconsistency (bukan hanya topical inconsistency), dibutuhkan layer evaluasi tambahan — misalnya entity extraction atau structured fact comparison. Ini adalah area pengembangan selanjutnya yang jelas arahnya.

---

## Finding 5: Latency sebagai Constraint Nyata

### Observasi
| Model | Context Length | Latency per Query |
|-------|----------------|-------------------|
| mistral | Panjang (~300 token) | ~320 detik |
| phi3:mini | Pendek (~50 token) | ~245 detik |
| phi3:mini | Sangat pendek | ~40 detik (via curl) |

Latency 4-5 menit per query pada CPU-only setup (16GB RAM) adalah constraint nyata yang membentuk cara framework ini digunakan.

### Keputusan Engineering
- Development: gunakan model ringan dengan context minimal untuk verifikasi pipeline
- Eval final: jalankan dengan Mistral dan dataset lengkap — satu kali, biarkan jalan
- Unit test: tidak pernah hit Ollama — semua test menggunakan jawaban hardcoded

### Implikasi
Framework ini dirancang untuk reproducibility, bukan speed. Siapa pun dengan laptop 16GB RAM dan koneksi internet untuk download model dapat mereproduksi semua hasil — tanpa GPU, tanpa API berbayar.

---

## Limitation yang Diakui

**1. Single embedding model.**  
Semua similarity calculation menggunakan nomic-embed-text. Hasil dapat berbeda dengan embedding model lain. Threshold yang dikalibrasi untuk nomic-embed-text mungkin perlu disesuaikan untuk model lain.

**2. Single LLM.**  
Evaluasi hanya dijalankan pada phi3:mini (development) dan mistral (target). Generalisasi ke model lain belum diverifikasi.

**3. Threshold dikalibrasi secara empiris.**  
Semua threshold (0.75, 0.65, 0.72, 0.70) ditentukan berdasarkan observasi dari dataset kecil (20-31 cases). Dataset yang lebih besar mungkin memerlukan re-kalibrasi.

**4. Domain terbatas.**  
Dataset hanya mencakup dokumentasi FastAPI. Behavior model pada domain teknis lain (misalnya dokumentasi database, protokol jaringan) belum dievaluasi.

**5. ROUGE-L sensitivity.**  
ROUGE-L sangat sensitif terhadap stopwords dan urutan kata. Jawaban yang semantically identik tapi menggunakan struktur kalimat berbeda akan mendapat ROUGE-L rendah.

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