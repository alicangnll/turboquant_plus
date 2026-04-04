# Bellek hedefleri: RSS ölçümü ve “kayıpsız” tanımı

Bu dosya TurboQuant+ / LLMTuning ile RAM optimizasyonu yaparken **ne ölçüleceğini** ve **başarıyı nasıl yorumlayacağınızı** sabitler.

## Birincil metrik: Peak RSS (fiziksel RAM baskısı)

- **Tanım:** Süreç için **tepe değer resident set size** — macOS’ta Activity Monitor’daki “Memory” veya komut satırında `/usr/bin/time -l` çıktısındaki `maximum resident set size` (bayt).
- **Neden:** LLMTuning (`MADV_DONTNEED` ile katman tahliyesi) ve düşük batch / düşük `-ngl` doğrudan **aynı anda RAM’de tutulan sayfa sayısını** etkiler; bu da peak RSS ile görülür.
- **KV tek başına:** Sadece KV önbelleği boyutunu hedefliyorsanız ikincil metrik olarak `ctx_len × katman × başlık × boyut` ve TurboQuant türüne göre sıkıştırma oranı kullanılabilir; üretimde yine **toplam süreç RSS** daha güvenilirdir.

## İkincil metrik: KV tahmini (yönlendirme)

Demo script içindeki `estimate_kv` (turbo4 ≈ 376, turbo2 ≈ 640, q8_0 ≈ 200 oranı) kabaca **KV’nin MiB karşılığını** verir; kesin değer kernel ve hizalamaya bağlıdır.

## “Kayıpsız” — bu projede kabul edilen anlam

| Bileşen | Gerçekçi beklenti |
|--------|---------------------|
| **GGUF ağırlıkları** | Zaten quant (ör. Q4_K_M); float orijinale göre **yaklaşık** temsil. |
| **TurboQuant KV** (`turbo4`, `turbo3`, …) | f16 KV’ye göre **sayısal olarak yaklaşık**; bit-bit aynı logits üretimi hedeflenmez. |
| **LLMTuning** | Doğru senkronizasyonda çıktıyı bozmaz; amaç fiziksel sayfa yönetimidir. |

**Ürün varsayılanı (kalite eşiği):** Aynı model ve makul prompt ile tutarlı, okunabilir yanıtlar (ör. hedef dilde anlamlı cevap). Daha sıkı doğrulama için ayrı perplexity / regresyon prompt seti kullanılabilir.

**Matematiksel “float16 KV + ağırlık ile birebir aynı çıktı”** ile **RAM’i yarıya indirmek** aynı anda genelde mümkün değildir; bilgi kaybı veya bağlam kısıtı gerekir.

## Bellek modları ile hedef (demo / `cli_config_export`)

| `mem_choice` | Etiket | RSS odağı | `batch` / `ubatch` (JSON) |
|--------------|--------|-----------|---------------------------|
| 1 | Performance | Daha yüksek peak RSS kabul; hız | 512 / 256 |
| 2 | Balanced | Orta RSS (~24 GB sınıfı hedef) | 256 / 128 |
| 3 | Ultra-Eco | Peak RSS’i düşük tut (16 GB sınıfı) | 32 / 32 |

`run_turboquant_demo_macos.sh` bu değerleri `tq_cli_config.json` dosyasına yazar ve oradan okur (her çalıştırmada güncellenir).

## Ölçüm prosedürü

Aynı model, aynı kısa prompt, iki farklı `mem_choice` veya farklı `--cache-type-*` ile:

1. [scripts/measure_rss_macos.sh](../scripts/measure_rss_macos.sh) kullanın **veya**
2. El ile: `/usr/bin/time -l your_llama_cli ... 2>&1 | grep maximum`

Karşılaştırmadan önce diğer ağır uygulamaları kapatın; ilk çalıştırmada disk önbelleği etkisi olabilir — mümkünse ikinci koşuyu da kaydedin.

**Önce / sonra örneği:** Aynı `*.gguf`, aynı `-p "..."` ve `-n`, sadece demo bellek seviyesini 2 ve 3 yapın (veya `--batch-size` / `-c` değiştirin); her koşuda `maximum resident set size` değerini not edin. Activity Monitor’da süreç seçiliyken “Memory” sütunu da kabaca karşılaştırma için kullanılabilir (anlık; peak değil).
