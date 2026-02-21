# Evaluation Analysis Report
## LLM Faithfulness & Consistency Framework — v1.1

**Models evaluated:** phi3:mini (development), mistral (primary eval)
**Dataset:** FastAPI documentation — 10 clean cases, 5 distractor cases, 6 paraphrase cases
**Infrastructure:** CPU-only, 16GB RAM, Ollama local inference
**Date:** February 2026

---

## Overview

Framework ini dirancang untuk menjawab satu pertanyaan yang jarang ditanyakan secara serius:

> *Apakah jawaban LLM benar-benar berasal dari dokumen yang diberikan, atau hanya terlihat benar karena model sudah melihat data serupa saat training?*

Laporan ini mendokumentasikan observasi empiris dari menjalankan framework terhadap data nyata — termasuk failure case yang tidak disembunyikan, limitation yang ditemukan selama development, dan implikasi dari setiap temuan.

---

## Hasil Evaluasi Lengkap

### Mistral — Clean Dataset

**Run ID:** 20260221_074512 | **Total waktu:** 160 menit | **Timeout:** 900s

| Case | Score | Keterangan |
|------|-------|------------|
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

**Avg faithfulness: 56.7% | Perfect: 5/10 | Failure rate: 50%**

---

### Mistral — Distractor Dataset

**Run ID:** 20260221_103528 | **Total waktu:** 120 menit

| Case | Score | Jawaban Model | Keterangan |
|------|-------|---------------|------------|
| distractor_001 | 0.0 | skip=5, limit=100 | Mengikuti nilai yang dimanipulasi |
| distractor_002 | 0.0 | POST | Mengikuti HTTP method yang dibalik |
| distractor_003 | N/A | (kosong) | Inference failed |
| distractor_004 | 0.0 | :filepath | Jawaban terlalu singkat |
| distractor_005 | 0.333 | 3 poin, 1 hallucination | Tambah "caching" yang tidak ada di konteks |

**Avg faithfulness: 8.3% | Perfect: 0/5 | Failure rate: 100%**

---

### Model Comparison — Mistral vs phi3:mini (Clean Dataset)

| Metrik | Mistral | phi3:mini |
|--------|---------|-----------|
| Avg faithfulness | **56.7%** | 46.7% |
| Perfect score (1.0) | 5/10 | 4/10 |
| Failure rate | 50% | 60% |
| False INSUFFICIENT_CTX | 2 | 2 |
| Avg latency per case | 463.7s | **222.9s** |
| Total eval time | 160 min | 77.7 min |

---

### Clean vs Distractor Comparison

| Metrik | Clean Dataset | Distractor Dataset |
|--------|--------------|-------------------|
| Avg faithfulness | 56.7% | 8.3% |
| Perfect score | 5/10 | 0/5 |
| Failure rate | 50% | 100% |
| False INSUFFICIENT_CTX | 2 | 0 |

---

## Findings

### Finding 1: False INSUFFICIENT_CONTEXT

**Observasi:**
Mistral menjawab `INSUFFICIENT_CONTEXT` untuk clean_001 dan clean_004 meskipun evidence ada di konteks dengan jelas. clean_004 adalah kasus ekstrem — similarity ground_truth ke konteks mencapai 0.9691 tapi model tetap menolak menjawab.

phi3:mini menunjukkan behavior yang sama pada dua case berbeda.

**Analisis:**
Ini bukan model yang jujur mengakui keterbatasan. Ini model yang gagal membaca konteks, lalu berlindung di balik escape hatch yang disediakan prompt. Framework versi pertama memberi score 1.0 untuk behavior ini — salah.

**Fix:**
Ditambahkan validasi dua tahap:
1. Jika model bilang `INSUFFICIENT_CONTEXT`, cek similarity antara `ground_truth` dan konteks.
2. Jika similarity ≥ 0.65, evidence ada → FALSE_INSUFFICIENT_CONTEXT → score 0.0.

**Threshold:**
- Evidence matching: **0.75** (klaim panjang)
- Insufficient validation: **0.65** (lebih rendah karena ground_truth lebih ringkas dari span konteks)

**Implikasi:**
Prompt yang terlalu agresif memberikan escape hatch mendorong model untuk under-answer. Ini tradeoff desain yang perlu dipertimbangkan di setiap RAG system.

---

### Finding 2: Claim Extraction Gagal pada Enumeration List

**Observasi:**
Jawaban seperti *"The OpenAPI schema includes API paths, parameters, authentication methods, response schemas, and rate limiting."* diekstrak sebagai satu klaim. Item yang tidak didukung konteks ikut lolos bersama item yang valid.

**Fix:**
`_expand_enumeration()` mendeteksi pola `"X includes A, B, C, and D"` dan memecahnya menjadi klaim atomik. Score berubah dari **1.0 → 0.6** untuk jawaban yang sama setelah fix.

**Implikasi:**
Granularitas claim extraction menentukan kualitas evaluasi. Evaluator yang mengevaluasi kalimat panjang sebagai satu unit akan banyak menghasilkan false positive.

---

### Finding 3: nomic-embed-text Menganggap Paraphrase sebagai Semantically Berbeda

**Observasi:**
Dua jawaban berikut dianggap semantically berbeda (similarity: 0.7677):
- `"By not declaring any default value for the parameter."`
- `"You can make it required by not providing a default value."`

Maknanya identik.

**Analisis:**
Ini limitation embedding model, bukan bug framework. nomic-embed-text dioptimalkan untuk document-level similarity, bukan sentence-level equivalence. Kalimat pendek dengan kata kerja berbeda menghasilkan embedding yang cukup berbeda.

**Fix:**
Threshold consistency diturunkan dari 0.80 ke **0.72** setelah observasi empiris. False positive masih tersisa untuk beberapa pasang paraphrase.

**Implikasi:**
Semantic similarity tidak cukup untuk mendeteksi semantic equivalence pada kalimat pendek dan padat. ROUGE-L memberikan sinyal komplementer yang penting.

---

### Finding 4: Semantic Similarity Tidak Mendeteksi Factual Inconsistency

**Observasi:**
`"PUT is used to update data"` vs `"PATCH is the correct HTTP method for partial updates"` — similarity **0.8337**, dianggap konsisten. Padahal PUT ≠ PATCH secara faktual.

**Analisis:**
Embedding model mendeteksi topik yang sama (HTTP methods untuk update), bukan fakta yang berbeda. Similarity topik ≠ similarity fakta.

**Implikasi:**
Untuk mendeteksi factual inconsistency, dibutuhkan layer tambahan — entity extraction atau structured fact comparison. Ini area pengembangan selanjutnya.

---

### Finding 5: Jawaban Terlalu Singkat — Blind Spot Evaluator

**Observasi:**
- clean_008: `'/files/home/johndoe/myfile.txt'` — benar secara faktual, score 0.0
- clean_009: `'PUT'` — benar secara faktual, score 0.0
- distractor_004: `':filepath'` — jawaban singkat, tidak evaluable

Claim extractor tidak bisa mengekstrak klaim dari path string atau single-word answers.

**Fix:**
Ditambahkan short answer path di `FaithfulnessEvaluator.evaluate()`:
- Jawaban < 5 kata → skip claim extraction
- Gunakan exact match bertingkat: exact → substring → character similarity
- Bonus check: apakah jawaban muncul di konteks (context grounding)
- Penalty 0.5x jika matched ke ground_truth tapi tidak grounded di konteks

**Implikasi:**
Semantic similarity tidak reliable untuk evaluasi single-token answers seperti HTTP methods atau file paths. Exact match jauh lebih tepat untuk kasus ini.

---

### Finding 6: Mistral Faithful ke Konteks yang Salah

**Observasi:**
Distractor dataset membuktikan: ketika konteks bilang `skip=5, limit=100`, model menjawab persis itu — meski nilai asli yang benar adalah `skip=0, limit=10`. Model tidak menggunakan training knowledge untuk koreksi.

Penurunan faithfulness 56.7% → 8.3% bukan karena model menjadi tidak faithful. Ini karena model sangat faithful ke konteks yang sudah dimanipulasi.

**Implikasi:**
RAG system yang well-behaved seharusnya mengikuti konteks yang diberikan. Tapi ini juga berarti: **garbage in, garbage out**. Kualitas retrieval sepenuhnya menentukan kualitas jawaban. Evaluasi retrieval pipeline sama pentingnya dengan evaluasi model.

---

### Finding 7: Mistral vs phi3:mini — Tradeoff Faithfulness vs Speed

**Observasi:**
Mistral lebih faithful (56.7% vs 46.7%) tapi 2x lebih lambat (463.7s vs 222.9s per case). Kedua model over-trigger INSUFFICIENT_CONTEXT dengan frekuensi yang sama (2 false insufficient masing-masing).

phi3:mini cenderung lebih verbose — clean_007 menghasilkan 7 klaim, semuanya semantically close tapi di bawah threshold. Model mencoba menjawab terlalu lengkap tapi tidak cukup grounded ke konteks.

**Implikasi:**
Untuk production: Mistral lebih aman untuk faithfulness. Untuk development iteration: phi3:mini 2x lebih cepat dengan tradeoff akurasi 10%. Pilihan model harus mempertimbangkan context ini.

---

### Finding 8: Timeout Tidak Konsisten pada CPU Inference

**Observasi:**
Dari run pertama, 6 dari 10 cases timeout setelah 600s. Tapi beberapa case berhasil selesai dalam 380-648s di run yang sama. Variabilitas ini tidak bisa diprediksi dari panjang konteks saja.

**Kemungkinan penyebab:** memory pressure saat pertama load, thermal throttling, variasi panjang output yang tidak bisa dikontrol.

**Fix:** Timeout dinaikkan ke 900s. Run kedua berhasil 10/10 tanpa failure.

---

## Limitation yang Diakui

**1. Single embedding model.**
Semua similarity calculation menggunakan nomic-embed-text. Threshold yang dikalibrasi mungkin perlu disesuaikan untuk embedding model lain.

**2. Dataset kecil.**
10 clean cases, 5 distractor cases. Threshold dikalibrasi dari observasi dataset kecil — mungkin perlu re-kalibrasi untuk dataset yang lebih besar.

**3. Domain terbatas.**
Hanya FastAPI documentation. Behavior model pada domain teknis lain belum diverifikasi.

**4. ROUGE-L sensitivity.**
Sangat sensitif terhadap stopwords dan urutan kata. Jawaban semantically identik dengan struktur berbeda akan mendapat ROUGE-L rendah.

**5. Short answer exact match belum divalidasi di semua edge cases.**
Implementasi baru — perlu lebih banyak data untuk mengkonfirmasi threshold yang tepat.

---

## Thresholds yang Dikalibrasi

| Threshold | Nilai | Tujuan |
|-----------|-------|--------|
| Faithfulness evidence matching | 0.75 | Klaim harus kuat match ke span konteks |
| Insufficient context validation | 0.65 | Ground truth lebih ringkas dari span konteks |
| Consistency semantic | 0.72 | Balance deteksi inkonsistensi vs variasi paraphrase |
| Consistency ROUGE-L | 0.40 | Paraphrase alami punya word overlap rendah |
| Robustness sensitivity | 0.85 | Perturbasi kecil tidak boleh ubah jawaban |
| Short answer char similarity | 0.85 | Fallback untuk jawaban yang hampir exact match |

---

## Kesimpulan

Framework ini tidak mengklaim bisa menilai "kualitas" jawaban LLM secara umum. Yang dilakukan lebih spesifik:

1. **Klaim dalam jawaban dapat di-trace ke dokumen sumber** — atau tidak.
2. **Jawaban stabil terhadap variasi formulasi pertanyaan** — atau tidak.
3. **Model bergantung pada konteks yang diberikan** — atau menjawab dari training memory.
4. **Jawaban tidak berubah karena perturbasi kecil pada prompt** — atau tidak robust.

Setiap keputusan dalam framework dapat dijelaskan secara logis. Setiap failure case dicatat, bukan disembunyikan. Setiap limitation diakui, bukan dihindari.

Ini bukan framework yang sempurna. Tapi ini framework yang jujur.