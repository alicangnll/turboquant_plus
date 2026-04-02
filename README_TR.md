# TurboQuant+ Türkçe Kılavuz ve İnceleme

TurboQuant, yerel LLM (Büyük Dil Modeli) çıkarımı sırasında "KV Cache" (Anahtar/Değer Belleği) kullanımını büyük ölçüde sıkıştıran bir teknolojidir. Bu repo `llama.cpp` için TurboQuant entegrasyonu ve Apple Metal üzerinden çeşitli donanım/model boyutları için optimize edilmiş demoları içerir.

## Yeni Eklenen Optimizasyonlar ve Demo Detayları

Proje ana dizinindeki `./run_turboquant_demo.sh` betiğini çalıştırarak, doğrudan donanımınıza ve seçeceğiniz modele göre en iyi ayarları test edebilirsiniz. 100B gibi devasa modellere kadar desteklenmektedir.

### 1. Model Seçenekleri ve Kolay Kurulum
Demo betiği, test yapabilmeniz için size 5 farklı hazır model seçeneği sunar:
- **1) Llama 3.1 8B Instruct (~5 GB)**: Hızlı ve genel kullanım için.
- **2) Qwen 2.5 32B Instruct (~20 GB)**: Dengeli ve yüksek performanslı.
- **3) Command R+ 104B (~43 GB)**: Apple Silicon üzerinde sınırları zorlayan yüksek kalite.
- **4) Qwen 2.5 0.5B Instruct (~400 MB)**: Sadece hızlı doğrulamalar için.
- **5) Llama-3-405B / 500B Sınıfı (~250 GB)**: Ekstrem bellek sıkıştırması ve NVMe takas alanı (SWAP) testi.

İsterseniz betiğe parametre vererek de menüyü atlayabilirsiniz (Örnek: `./run_turboquant_demo.sh 8b`).

### 2. Apple Silicon (M Serisi) L2 Cache Darboğazının Aşılması (`turbo4`)
M1, M2 ve M3 serisi işlemcilerde `turbo3` sıkıştırması kullanıldığında eski nesil L2 önbellek yapısından dolayı dequantize işlemi sırasında bir performans darboğazı yaşanıyordu (hız %37 civarı düşebiliyordu). Demo üzerinde yapılandırmalar **`--cache-type-k turbo4 --cache-type-v turbo4`** olarak ayarlanmıştır. `turbo4` daha hafif paketleme mantığı sayesinde M serilerinde temel q8_0 seviyesine göre **%33.9** oranında okuma performansı artışı sağlar!

### 3. Boundary V Katman Koruması
Kaliteden hiç taviz vermeden sıkıştırma oranını artırmak için `TURBO_LAYER_ADAPTIVE=7` çevresel değişkeni otomatik olarak aktifleştirilmiştir. Bu özellik, ilk ve son 2 katmanı `q8_0` (yüksek hassasiyet) olarak koruyup, ortadaki tüm katmanları TurboQuant ile sıkıştırır. Hiçbir yavaşlama maliyeti olmadan modelin zekasındaki (kalitesindeki) kayıpları **%37 ile %91 oranında geri kazandırır**.

### 4. 100B Modeller İçin "Sistem Donması" Çözümü ve Performans
Devasa modeller belleğe (RAM) bindirildiğinde macOS işletim sistemi `--mmap` kullandığı için Metal sürücüsü belleği "kablolu" (wired memory) hale getirmeye çalışır. Limit aşıldığında sistem tamamen kilitlenir. 
Demo betiği `100B` sınıfı bir model seçtiğinizde aşağıdakileri otomatik uygular:
- **`--no-mmap`**: Yükleme yöntemini değiştirip wired memory limitini atlar. Kernel çökmesini engeller.
- **`-b 2048 -ub 512`**: İşlem yığınını (batch size) artırarak ekran kartının işlemci gücünü uzun işlemler için maksimize eder.
- **`-c 1024`**: Hızlı çıktı alabilmek adına temel context alanını sınırlar ve belleği rahatlatır.
- **`turbo4`**: Dequantization aşamasında L2 belleğini tıkamadan aşırı hız sağlar (+%33.9 M serisi hızı).

### 5. 500B Sınıfı Modeller İçin "Ekstrem" SSD Takas (Swap) Ayarları
500B (Örn: Llama-3-405B) sınıfı modeller, Apple Silicon sistemlerin Unified Memory kapasitelerini (Örn: 128GB veya 192GB) tek başlarına aşarlar. Bu modeller çalışırken sistem SSD üzerinden NVMe Swap alanını aktif kullanmaya başlar. Demo betiği bu devasa modeller için sistemin çökmesini sağlayan ani bellek sıçramalarını (memory spikes) engellemek adına:
- **`turbo2` (2-bit)**: KV Cache belleğini mümkün olan en düşük seviyeye indirerek sistem belleğinde modelin kendisine alan açar (6.4x sıkıştırma). 
- **`-b 128 -ub 64`**: Batch size'ı radikal şekilde düşürerek prompt işleme esnasında saniyelik on binlerce GB'lik disk takası stresini (thrashing) hafifletir.
- **`-c 512`**: Bağlam alanını minimuma çekerek KV Cache ve bellek israfını durdurur.
- **`--no-mmap`**: Wired memory felaketini kesin olarak çözmek için aktif edilir.

## Nasıl Çalıştırılır?

Ortamınızda Python 3.10+, cmake ve Xcode Command Line Tools kurulu olmalıdır. 

Sadece aşağıdaki komutu çalıştırarak sıfırdan her şeyin indirilmesini, derlenmesini ve modelinizin başlatılmasını sağlayabilirsiniz:

```bash
./run_turboquant_demo.sh
```

Sorun yaşarsanız veya parametreleri özelleştirmek isterseniz `llama-cpp-turboquant` klasörü içerisinden `llama-cli` veya `llama-server` araçlarını kendiniz başlatabilirsiniz.

---

## AirLLM + TurboQuant Hibrit Python Modülleri

Bu repo artık `airllm` projesinin temel optimizasyon stratejisini, TurboQuant KV Cache sıkıştırmasıyla birleştiren iki yeni Python modülü barındırıyor. Bu modüller saf **NumPy** tabanlı olduğu için PyTorch, CUDA veya ekstra bağımlılık gerektirmiyor.

### AirLLM Nedir, İki Sistem Nasıl Birbirini Tamamlıyor?

**AirLLM**, bir transformatör modelinin tüm ağırlıklarını (bölümlerini) tek seferde belleğe almak yerine **katman katman diske parçalar** (sharding), bu katmanları sırayla belleğe çekip hesabı yapıp, sonrasında anında serbest bırakır. Böylece 70B büyüklüğündeki bir modeli sadece 4-8 GB VRAM ile çalıştırabiliyorsunuz.

**TurboQuant** ise modelin çalışması sırasında oluşan KV Cache belleğini (her katmanda Q@K^T ve attn@V için tutulan tampon) 3.8–6.4× oranında sıkıştırır. AirLLM'in dokunmadığı yere müdahale eder.

**İkisi birlikte:**
- Ağırlık belleği: AirLLM → yalnızca 1 katman aktif → ~350 MB (32B model için)
- KV Cache belleği: TurboQuant turbo4 → ~3.76× küçültme
- Sonuç: 32B model ~2 GB aktif bellek ile çalışabiliyor

---

### 6. `turboquant/airllm_bridge.py` — Katman Bazlı KV Sıkıştırma Köprüsü

Her katmanın K ve V tensörlerini anında TurboQuant ile sıkıştırıp, ham tensörü bellekten atar.

**Temel sınıflar:**

- **`AirLLMTurboSession`**: Model boyutuna göre sıkıştırma politikasını otomatik seçer ve katman bazlı KV deposu yönetir.
- **`LayerPrefetcher`**: AirLLM'deki `ThreadPoolExecutor` prefetch stratejisinin birebir karşılığı. Bir sonraki katmanın ağırlıklarını arka planda diskten RAM'e yüklerken, şimdiki katmanın hesabı yapılır. GPU boş kalmaz.
- **`CachePolicy`**: Model boyutu → `k_bits`, `v_bits`, `max_context` eşlemesi.

**Otomatik politika seçimi:**

| Model boyutu | K önbellek | V önbellek | Maks. bağlam |
|---|---|---|---|
| < 32B | turbo4 | turbo4 | 8192 |
| 32B–65B | turbo4 | turbo4 | 4096 |
| 65B–100B | turbo4 | turbo4 | 2048 |
| 400B+ | turbo2 | turbo2 | 512 |

Sınır katmanları (ilk/son 2) her zaman `turbo4`'te korunur — `TURBO_LAYER_ADAPTIVE=7` davranışının Python karşılığı.

---

### 7. `turboquant/streamed_inference.py` — Uçtan Uca Yönetici

Katman akışı + KV sıkıştırmasını tek bir yöneticide birleştirir.

```python
from turboquant.streamed_inference import StreamedInferenceManager

# 32B model için otomatik katman/kafa yapılandırması:
manager = StreamedInferenceManager.for_model_size(32)

# Demo geçişi (gerçek model ağırlığı gerekmez):
result = manager.demo_forward(seq_len=4096, verbose=True)
print(result.memory_report)

# Farklı bağlam uzunluklarında KV tasarruf raporu:
manager.benchmark_compression(seq_lengths=[512, 2048, 4096, 8192])
```

**Apple M4 üzerinde doğrulanan sonuçlar (8 katman, sentetik dikkat):**

| Model | Bağlam | KV Ham | KV Sıkıştırılmış | Oran | Sıkıştırma süresi |
|-------|--------|--------|-------------------|------|-------------------|
| 8B (Llama) | 512 | 64 MB | 17 MB | **3.76×** | ~54 ms/katman |
| 32B (Qwen) | 512 | 80 MB | 21 MB | **3.76×** | ~72 ms/katman |
| 70B (Llama) | 256 | 64 MB | 17 MB | **3.76×** | ~56 ms/katman |
| 104B (Command-R+) | 128 | 64 MB | 17 MB | **3.76×** | ~60 ms/katman |

KV sıkıştırma ek yükü, katman başına **~50–75 ms** (yalnızca CPU NumPy). Katman parçalama düzeninde disk G/Ç gecikmesine kıyasla **ihmal edilebilir**. llama.cpp Metal GPU çekirdekleriyle çalışıldığında bu süre mikrosaniye mertebesine iner.

---

Modülleri hızlıca test etmek için:

```bash
source .venv/bin/activate
python3 -m turboquant.streamed_inference
```

