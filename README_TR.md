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
