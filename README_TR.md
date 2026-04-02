# TurboQuant+ (Türkçe Kılavuz)

> ### [Başlangıç Rehberi](docs/getting-started.md) | [Yapılandırma Önerileri](docs/turboquant-recommendations.md) | [llama.cpp Çatallaması](https://github.com/TheTom/llama-cpp-turboquant)

[TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026) algoritmasının gelişmiş bir uygulamasıdır. Yerel LLM çıkarımı (inference) için KV önbelleği (KV cache) sıkıştırmasına odaklanır ve orijinal makalenin ötesinde deneysel bulgular, performans optimizasyonları ve Apple Silicon özelinde iyileştirmeler içerir.

## Projenin Amacı

Bu depo, `llama.cpp` için deneysel bir araştırma ve entegrasyon çalışma alanıdır. Amacımız; farklı donanımlarda karşılaştırılabilir kıyaslama verileri toplamak, KV sıkıştırma yaklaşımlarını doğrulamak ve kararlı hale gelen parçaları kademeli olarak ana `llama.cpp` projesine eklemektir.

## 🚀 2026 Motor Güncellemeleri: Stabilite ve Bellek Yönetimi

Yeni nesil TurboQuant+ motoru, Apple Silicon cihazlarda (8GB-24GB RAM) 32B'den 500B'ye kadar olan modellerin sistem kilitlenmeden çalıştırılabilmesi için kritik özellikler sunar:

### 1. Bellek Optimizasyon Seviyeleri
Başlangıçta üç farklı mod seçilebilir:
- **Performance**: Maksimum GPU kullanımı. 32GB+ RAM sistemler için en hızlı seçenek.
- **Balanced**: 24GB RAM (M-Pro serisi) için optimize edilmiştir. GPU bütçesinin %30-40'ını kullanır.
- **Ultra-Eco**: 8-16GB RAM için güvenli mod. Minimal GPU, `turbo2` (2-bit) KV önbelleği ve 512 context limiti.

### 2. Çift İvmelendirme: Metal + OpenMP
Motor artık Apple Metal (GPU) ve OpenMP (CPU) donanımlarını aynı anda kullanır:
- **GPU**: Transformer katmanlarını ve KV aritmetiğini işler.
- **CPU**: Performans Çekirdekleri (P-Cores), arka planda KV sıkıştırma ve Walsh-Hadamard rotasyonları için izole edilir. Bu sayede arayüz takılmaları engellenir ve toplam işlem hızı artar.

### 3. Hibrit KV Önbelleği (`turbo4` + `turbo2`)
Key (Anahtar) ve Value (Değer) tensörleri için bağımsız hassasiyet sunar. **K-cache**'in 4-bit (`turbo4`) olarak tutulması modelin zekasını korurken, **V-cache**'in 2-bit (`turbo2`) olarak sıkıştırılması bellek kullanımını dramatik şekilde düşürür.

### 4. Akıllı MMAP Stratejisi
Fiziksel RAM'den büyük modeller (100B, 500B) için geliştirilen akıllı bellek eşleme sayesinde, RAM yetmediğinde sistemin SSD üzerinden **NVMe Swap** kullanarak kilitlenmeden çalışması sağlanır.

### 5. Evrensel Linux Desteği (CUDA & ROCm)
TurboQuant+ artık Linux sunucuları ve iş istasyonları için tamamen optimize edilmiştir. Motor otomatik olarak şunları algılar:
- **NVIDIA GPU**: Maksimum verim için CUDA arka ucunu etkinleştirir.
- **AMD GPU**: Yüksek performanslı açık kaynaklı GPU ivmelendirmesi için ROCm/HIP arka ucunu etkinleştirir.
- **CPU (OpenMP)**: GPU olmayan sistemlerde AVX/AMX komut setlerini kullanarak hızlı CPU çıkarımı sağlar.

## Temel Bulgular ve İnovasyonlar

TurboQuant+ geliştirme sürecinde doğrulanan üç temel bulgu:

1.  **V sıkıştırması "bedavadır":** Value (Değer) önbelleğinin 2-bit'e kadar sıkıştırılması, Key (Anahtar) hassasiyeti korunduğu sürece model kalitesini bozmaz.
2.  **Kalite kaybının kaynağı K sensörüdür:** Asimetrik yapılandırmalar (q8_0-K + turbo-V), zekayı korurken yüksek sıkıştırma sağlar.
3.  **Sınır Katmanları Hassastır (Boundary V):** İlk 2 ve son 2 katmanın yüksek hassasiyette (q8_0) korunması, kalite kaybının %37-91'ini geri kazandırır. `TURBO_LAYER_ADAPTIVE=7` ile aktif edilir.

## Performans ve Kalite (M5 Max 128GB)

| Önbellek Tipi     | Bit/Değer | Sıkıştırma | PPL (Kalite) | q8_0 Kıyas |
| ----------------- | --------- | ---------- | ------------ | ---------- |
| q8_0 (Varsayılan) | 8.5       | 1.9x       | 6.111        | Referans   |
| **turbo4**        | **4.25**  | **3.8x**   | **6.125**    | **+0.23%** |
| turbo3            | 3.5       | 4.6x       | 6.176        | +1.06%     |
| turbo2            | 2.5       | 6.4x       | 6.507        | +6.48%     |

> **Not:** M1, M2 ve M3 işlemcilerde `turbo4`, eski nesil L2 önbellek darboğazını aşarak q8_0'a göre **%33.9** daha hızlı okuma performansı sağlar.

## LLMTuning + TurboQuant Hibrit Pipeline

32B ve üzeri devasa modelleri Apple Silicon üzerinde çalıştırmak için LLMTuning'in "katman parçalama" stratejisi ile TurboQuant birleştirilmiştir. **3 aşamalı asenkron pipeline** yapısı şöyledir:

1.  **Aşama 1 (Disk I/O)**: `LayerPrefetcher` bir sonraki katmanı RAM'e yükler.
2.  **Aşama 2 (GPU Hesaplama)**: Aktif katman GPU üzerinde çıkarım yapar.
3.  **Aşama 3 (CPU Sıkıştırma)**: `KVCompressionWorker` bir önceki katmanın KV önbelleğini CPU çekirdeklerinde arka planda sıkıştırır.

Bu yöntemle 32B bir model, sadece **~2-4 GB aktif RAM** ile çalıştırılabilir.

### GPT-OSS-20B (MoE) Desteği
`openai/gpt-oss-20b` sınıfı tarafından kullanılan **Mixture of Experts (MoE)** mimarisi için tam destek entegre edilmiştir. Demo betiklerimiz, 24 katmanlı MoE parametrelerini (8 KV kafası) bu modeller için belleği optimize edecek şekilde otomatik olarak yapılandırır.

## Nasıl Çalıştırılır?

### Gereksinimler
- Python 3.10+
- cmake ve C++ derleyici
- Xcode Command Line Tools (macOS)

### Hızlı Başlangıç (Demo)
```bash
git clone https://github.com/TheTom/turboquant_plus.git
cd turboquant_plus
# macOS: Metal + OpenMP için optimize edildi
./run_turboquant_demo.sh

# Linux: CUDA + ROCm + OpenMP için optimize edildi
./run_turboquant_demo_linux.sh

# Windows: CUDA + OpenMP için optimize edildi
run_turboquant_demo.bat
```

### llama.cpp ile Kullanım
Sunucu (server) veya CLI modunda `--cache-type-k turbo4 --cache-type-v turbo4` parametrelerini ekleyerek TurboQuant aktif edilebilir.

---

**Destek:** Bu projeyi beğendiyseniz [GitHub Sponsors](https://github.com/sponsors/TheTom) üzerinden destek olabilirsiniz.

**Lisans:** Apache License 2.0. Copyright 2026 Tom Turney.
