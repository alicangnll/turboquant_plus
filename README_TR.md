# TurboQuant+ Türkçe Kılavuz ve İnceleme

TurboQuant, yerel LLM (Büyük Dil Modeli) çıkarımı sırasında "KV Cache" (Anahtar/Değer Belleği) kullanımını büyük ölçüde sıkıştıran bir teknolojidir. Bu repo `llama.cpp` için TurboQuant entegrasyonu ve Apple Metal üzerinden çeşitli donanım/model boyutları için optimize edilmiş demoları içerir.

## Yeni Eklenen Optimizasyonlar ve Demo Detayları

Proje ana dizinindeki `./run_turboquant_demo.sh` betiğini çalıştırarak, doğrudan donanımınıza ve seçeceğiniz modele göre en iyi ayarları test edebilirsiniz. 

### 1. Model Seçenekleri ve Kolay Kurulum
Demo betiği, test yapabilmeniz için size 5 farklı hazır model seçeneği sunar:
- **1) Llama 3.1 8B Instruct (~5 GB)**: Hızlı ve genel kullanım için.
- **2) Qwen 2.5 32B Instruct (~20 GB)**: Dengeli ve yüksek performanslı.
- **3) Command R+ 104B (~43 GB)**: Apple Silicon üzerinde sınırları zorlayan yüksek kalite.
- **4) Qwen 2.5 0.5B Instruct (~400 MB)**: Sadece hızlı doğrulamalar için.
- **5) Llama-3-405B / 500B Sınıfı (~250 GB)**: Ekstrem bellek sıkıştırması ve NVMe takas alanı (SWAP) testi.

### 2. Bellek Optimizasyon Seviyeleri (VRAM & RAM Kontrolü)
Artık her model için üç farklı çalışma modu seçebilirsiniz, böylece 24 GB veya 16 GB sistemlerde bile 32B/70B modellerini çalıştırabilirsiniz:
- **1) Performance**: Maksimum GPU kullanımı, hızlı çıkarım. 32GB+ RAM önerilir.
- **2) Balanced**: 24GB RAM (M-Pro serisi) için optimize edilmiştir. 512-1024 context.
- **3) Ultra-Eco**: 8-16GB RAM için en güvenli moddur. Hem K hem V önbelleği 2-bit'e (`turbo2`) düşürülür, context 512 ile sınırlandırılır.

### 3. Devasa Modeller İçin "Hibrid" ve "Smart MMAP" Çözümü
32B, 100B ve 500B gibi devasa modellerde sistem donmalarını ve Metal budget hatalarını engellemek için script otomatik olarak şunları uygular:
- **Hibrid Cache**: K-cache `turbo4` (zeka koruması), V-cache `turbo2` (bellek tasarrufu) olarak ayrılır.
- **Auto-NGL**: Apple Metal sürücüsünün `recommendedMaxWorkingSetSize` limiti anlık okunarak en güvenli GPU katman sayısı otomatik hesaplanır.
- **Smart MMAP**: `--no-mmap` yerine akıllı bellek eşleme kullanılarak RAM yetmediğinde sistemin SSD Swap üzerinden kilitlenmeden çalışması sağlanır.

### 4. Apple Silicon L2 Cache Darboğazının Aşılması (`turbo4`)
M1, M2 ve M3 serisi işlemcilerde `turbo3` sıkıştırması kullanıldığında eski nesil L2 önbellek yapısından dolayı darboğaz yaşanabiliyordu. Demo üzerinde yapılandırmalar **`--cache-type-k turbo4 --cache-type-v turbo4`** olarak ayarlanmıştır. `turbo4` M serilerinde temel q8_0 seviyesine göre **%33.9** oranında okuma performansı artışı sağlar!

---

## AirLLM + TurboQuant Hibrit Python Modülleri

Bu repo artık `airllm` projesinin temel optimizasyon stratejisini, TurboQuant KV Cache sıkıştırmasıyla birleştiren yeni Python modülleri barındırıyor.

- **AirLLM**: Modeli katman katman diske parçalar (sharding), aktif katman bittiğinde anında serbest bırakır.
- **TurboQuant**: KV Cache belleğini 3.8–6.4× oranında sıkıştırır.
- **Sonuç**: 32B model, bu yöntemle sadece **~2-4 GB aktif RAM** ile çalışabiliyor.

---

## Nasıl Çalıştırılır?

Ortamınızda Python 3.10+, cmake ve Xcode Command Line Tools kurulu olmalıdır. 

```bash
./run_turboquant_demo.sh
```

Sorun yaşarsanız veya parametreleri özelleştirmek isterseniz `llama-cpp-turboquant` klasörü içerisinden `llama-cli` veya `llama-server` araçlarını kendiniz başlatabilirsiniz.
