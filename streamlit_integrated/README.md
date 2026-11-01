# Ketika Bumi Berguncang — Streamlit

Integrasi dari `story.html` terbaru yang diunggah: animasi desa → tanah terbelah pada step 6 → tujuh grafik Plotly di dalam jurang.

## Jalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Masuk ke halaman aplikasi, lalu **scroll di dalam komponen halaman** untuk mengikuti animasi dan membaca ketujuh grafik. Aplikasi menggunakan `data/katalog_gempa.csv` dari dataset Kaggle `kekavigi/earthquakes-in-indonesia` agar hasil dapat dijalankan tanpa koneksi internet.

## Struktur

- `app.py` — membaca HTML, menghasilkan figure dari Python, mengisi marker `<!--VIZ_01_START--> ... <!--VIZ_07_END-->`, dan menampilkan halaman via `components.html`.
- `story.html` — **file HTML terbaru pengguna**, dengan animasi pembuka, retakan, dinding jurang, section grafik, dan pengendali scroll.
- `visualizations.py` — data cleaning dan tujuh figure Plotly.
- `data/katalog_gempa.csv` — salinan dataset yang disertakan untuk reproduksibilitas.
- `preview_story.html` — output statis lengkap dengan tujuh plot yang dapat dibuka di browser biasa (relatif besar karena menyertakan Plotly.js dan data figure).

## Batasan

- Angka magnitudo dan jam animasi opening bersifat ilustratif, bukan rekonstruksi kejadian spesifik.
- Diagram kedalaman yang mengikuti scroll di jurang bersifat ilustrasi, bukan kedalaman episentrum aktual.
- Grafik Palu menunjukkan catatan yang memenuhi radius/waktu, bukan identifikasi aftershock secara ilmiah.
- Kode visualisasi berasal dari versi integrasi sebelumnya, tidak otomatis identik dengan seluruh pengaturan terakhir di notebook.
- UI scroll dan animasi perlu diuji manual melalui browser/Streamlit untuk memastikan tampilan mobile.
