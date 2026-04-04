# TurboQuant+ (Türkçe Kılavuz)

> ### [Başlangıç Rehberi](docs/getting-started.md) | [Yapılandırma Önerileri](docs/turboquant-recommendations.md) | [llama.cpp Çatallaması](https://github.com/TheTom/llama-cpp-turboquant)

[TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026) algoritması ve **LLMTuning** orkestrasyonunun gelişmiş bir uygulamasıdır. Yerel LLM çıkarımı için KV önbelleği sıkıştırması ile akıllı bellek yönetimini birleştirerek devasa modellerin tüketici donanımlarında çalışmasını sağlar.

## Projenin Amacı

Bu depo, `llama.cpp` için deneysel bir araştırma ve entegrasyon çalışma alanıdır. Amacımız; farklı donanımlarda karşılaştırılabilir kıyaslama verileri toplamak, KV sıkıştırma yaklaşımlarını doğrulamak ve kararlı hale gelen parçaları kademeli olarak ana `llama.cpp` projesine eklemektir.

## 🚀 2026 Motor Teknolojileri: Ayrıştırılmış Mükemmeliyet

TurboQuant+ ve LLMTuning, en verimli çıkarım deneyimini sunmak için birlikte çalışır.

### 🧩 LLMTuning: Akıllı Bellek Orkestrasyonu
LLMTuning, donanım kaynaklarının aşırı bellek baskısı altında bile stabiliteyi sağlayacak şekilde yönetilmesinden sorumlu "beyin" katmanıdır.

- **Donanım Farkındalıklı NGL Otomasyonu**: Metal `recommendedMaxWorkingSetSize` gibi platform bütçelerini otomatik algılar ve en güvenli `-ngl` katman sayısını hesaplar.
- **Aktif Parçalama (Active Sharding) & Akıllı MMAP**: Fiziksel RAM'den büyük modeller (100B+) için `madvise` kullanarak işlenen katmanları anında bellekten tahliye eder, NVMe Swap ile sistem kilitlenmeden çalışır.
- **Ultra-Eco Optimizasyonu**: Mükerrer bellek atamalarını (Repack Suppression) engelleyerek 8B modellerin **~1.1GB toplam RAM** ile çalışabilmesini sağlar.
- **Performans Modları**: Başlangıçta *Performance* (Maks GPU), *Balanced* (Güvenli VRAM) veya *Ultra-Eco* (Düşük RAM) modları seçilebilir.


### ⚡ TurboQuant+: Sıkıştırma Çekirdeği
TurboQuant+, modelin çalışma belleğini küçülten gerçek sıkıştırmayı sağlayan yüksek hızlı motor katmanıdır.

- **PolarQuant Sıkıştırma**: 2-bit (`turbo2`), 3-bit (`turbo3`) ve 4-bit (`turbo4`) KV önbellek sıkıştırması ile sıfıra yakın kalite kaybı.
- **Çift İvmelendirme**: Transformer hesaplamaları için **Metal (GPU)**, rotasyon işlemleri için **OpenMP (CPU)** bir arada kullanılır.
- **Hibrit KV Önbelleği**: K (4-bit) ve V (2-bit) tensörleri için bağımsız hassasiyet sunarak zeka ve tasarruf dengesini kurar.
- **Sparse V Optimizasyonu**: Düşük ağırlıklı V pozisyonlarını atlayarak, uzun bağlamlarda (long context) okuma hızını **%22.8**'e kadar artırır.
- **Sınır Koruması (Boundary Protection)**: Bağlam tutarlılığını korumak için ilk/son katmanları yüksek hassasiyette tutar.


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

### 🚀 Stabil Asenkron Pipeline (`--turbo-async`)
Motor artık gerçek bir 3 aşamalı asenkron orkestrasyon kullanır:
- **Önyükleme (LLMTuning)**: GPU çalışırken bir sonraki adımın ağırlıkları diskten RAM'e çekilir.
- **Hesaplama (LLM Engine)**: Metal GPU çekirdekleri üzerinde tam hızda çıkarım yapılır.
- **Sinyalleşme (TurboQuant)**: KV önbellek bütünlüğünü bozmadan güvenli arka plan yönetimi sağlanır.
- **Sonuç**: Llama 3.1 8B modelinde **943.7 t/s** gibi rekor prompt hızları.

> [!TIP]
> Daha fazla mimari detay, bileşen haritası ve veri akış diyagramı için [MAP.md](MAP.md) dosyasına göz atabilirsiniz.

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
