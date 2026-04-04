# 🚀 TurboQuant+ // LLMTuning
### 2026 Yerel LLM Çıkarımı İçin Ekstrem Verimlilik Motoru

> **TurboQuant+** ve **LLMTuning**, devasa dil modellerini (32B–500B) tüketici sınıfı donanımlarda (16GB–64GB RAM) hız ve stabiliteden ödün vermeden çalıştırmak için tasarlanmış ikiz teknolojilerdir.

[Mimari Haritası](MAP.md) | [Geliştirme Yol Haritası](PLAN.md) | [İngilizce Rehber (English)](README.md)

---

## 🌟 Vizyon: Uç Birimlerde Devasa Güç
Çoğu yerel çıkarım motoru, fiziksel RAM'den büyük modelleri çalıştırmakta zorlanır. Bu proje, modelin **Matematiksel Temsili** (TurboQuant+) ile **Donanım Orkestrasyonunu** (LLMTuning) birbirinden ayırarak bu sorunu çözer.

*   **TurboQuant+**: ICLR 2026 makalesine dayanan, kayıpsıza yakın 2-bit, 3-bit ve 4-bit KV önbellek sıkıştırması sunan son teknoloji bir sıkıştırma çekirdeğidir.
*   **LLMTuning**: Model belleğini sanallaştıran, katmanları NVMe SSD ve RAM arasında engelleyici olmayan (non-blocking) asenkron bir boru hattında (pipeline) düzenleyen akıllı bir orkestrasyon katmanıdır.

---

## 🧩 Temel Direk 1: LLMTuning (Orkestrasyon)
LLMTuning, LLM'niz için bir "işletim sistemi" görevi görür. Donanım kaynaklarını yöneterek, 104B'lik bir modeli 16GB'lık bir MacBook'ta çalıştırırken bile motorun asla OOM (Bellek Yetersizliği) durumuna düşmemesini sağlar.

*   **Aktif Parçalama (Active Sharding - `madvise`)**: GPU hesaplamayı bitirir bitirmez dönüştürücü (transformer) katmanlarını fiziksel RAM'den otomatik olarak tahliye eder. RAM yalnızca *aktif* katmanı ve bir önbelleği tutar.
*   **Yerel Bütçe Keşfi**: Apple Metal `recommendedMaxWorkingSetSize` gibi platform bellek sınırlarını C++ `sysctl` çağrıları ile otomatik algılar ve en uygun GPU katman sayısını (`-ngl`) hesaplar.
*   **Soğuk Başlatma Tahliyesi**: Model ağırlıklarını hemen SSD destekli sanal belleğe taşıyarak başlangıçtaki RAM sıçramalarını minimize eder; bu sayede 8B modeller **~1.1GB RAM izi** ile başlayabilir.
*   **Öngörülü Sayfalama**: GPU mevcut katmanla meşgulken, bir sonraki katmanı NVMe'den RAM'e yüklemeye başlamak için `MADV_WILLNEED` ipuçlarını kullanır.

---

## ⚡ Temel Direk 2: TurboQuant+ (Sıkıştırma)
TurboQuant+, modelin çalışma belleğini (KV Cache) 6.4 katına kadar küçülten yüksek hızlı sıkıştırma sağlar.

*   **PolarQuant (2/3/4-bit)**: Dikgen Walsh-Hadamard rotasyonlarını kullanarak dikkat (attention) tensörlerini bir Beta dağılımına dönüştürür; bu, neredeyse sıfır kalite kaybıyla optimal skaler kuantizasyona olanak tanır.
*   **Seyrek V (Sparse V) Optimizasyonu**: Düşük öncelikli Değer (Value) tensörlerini atlayan dikkat kapılı bir dekuantizer kullanarak uzun bağlamlı (long-context) okuma hızlarını **~%22.8** artırır.
*   **Çift İvmelendirme**: Transformer matematiğini **GPU (Metal/CUDA)** üzerinde çalıştırırken, rotasyon ve kuantizasyon işlemlerini eş zamanlı olarak **CPU (OpenMP)** üzerinde gerçekleştiren paralel bir hesaplama modelidir.
*   **Sınır Koruması**: Uzun menzilli tutarlılığı ve formatı korumak için ilk ve son katmanları yüksek hassasiyette (Örn: Q8_0) tutar.

---

## 🚀 3 Aşamalı Asenkron Boru Hattı
Motor, **Turbo-Async Pipeline** adı verilen özel bir engelleyici olmayan düzenleme uygular:

1.  **Aşama 1: Ön Yükleme (LLMTuning)**: Aşama 2 çalışırken Katman N+1'i diskten RAM'e yükler.
2.  **Aşama 2: Hesaplama (LLM Motoru)**: GPU, yerel çekirdekleri kullanarak mevcut Katman N'yi yürütür.
3.  **Aşama 3: Sıkıştırma (TurboQuant+)**: CPU, Aşama 2 biter bitmez Katman N için KV önbelleğini sıkıştırır.

![Asenkron Boru Hattı Görselleştirmesi](file:///Users/alicangonullu/.gemini/antigravity/brain/2e148d66-421c-432a-b624-209399070db4/media__1775313466809.png)

**Performans Sonucu**: Llama 3.1 8B ile sıfır "@@@@" bozulması ve **943.7 t/s**'ye varan komut işleme hızı.

---

## 🛠️ Hızlı Başlangıç

### 1. Gereksinimler
*   **macOS**: Xcode Komut Satırı Araçları, Python 3.10+, [libomp](https://formulae.brew.sh/formula/libomp) (Homebrew üzerinden).
*   **Linux**: GCC/Clang, CMake, OpenMP.
*   **Windows**: MSVC, CMake.

### 2. Derleme
```bash
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

### 3. Demoyu Çalıştırın
Sıfır konfigürasyon deneyimi için donanıma optimize edilmiş betikler sunuyoruz:
*   **macOS**: `./run_turboquant_demo_macos.sh`
*   **Linux**: `./run_turboquant_demo_linux.sh`
*   **Windows**: `run_turboquant_demo.bat`

---

**Lisans**: Apache 2.0. Telif Hakkı 2026 Tom Turney.
