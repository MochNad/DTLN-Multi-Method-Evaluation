# 🎵 DTLN Multi-Method Audio Evaluation

**Full-stack audio processing system** untuk evaluasi metode denoising menggunakan DTLN (Deep Transfer Learning Noise Suppression) dengan **web app Flask** yang modern dan **Jupyter notebook** untuk batch processing.

## ✨ Fitur Utama

### 🌐 Web App (Flask)

- **Modern Responsive UI**: Tailwind CSS dengan shadcn/ui theme
  - Mobile-first design (1 col → 2 col tablet → 4 col desktop)
  - Dark mode ready dengan color palette slate/blue/purple/amber/rose
- **3 Metode Processing**:
  - 🎯 **Deterministik**: Fixed noise + DTLN (reproducible, requires uploaded noise)
  - 🎲 **Stokastik**: Random noise + DTLN (robustness testing, synthetic noise)
  - 🔧 **Tradisional**: Spectral Subtraction / Wiener Filter (DSP baseline)
- **Audio Processing**:
  - Upload clean audio + noise audio
  - Adjustable SNR (Signal-to-Noise Ratio)
  - Real-time processing dengan ONNX Runtime
  - Built-in audio players untuk preview
- **Quality Metrics** (4 decimal precision):
  - STOI (Speech Intelligibility) - 0.0000 to 1.0000
  - PESQ (Perceptual Evaluation) - -0.5000 to 4.5000
  - MSE (Mean Squared Error) - lower is better
  - MRE (Mean Relative Error) - lower is better
- **Visualisasi**:
  - Combined 2x2 spectrogram (Clean, Noise, Mixed, Processed)
  - Magma colormap (gamma style, not green/viridis)
  - High-resolution matplotlib rendering (10 DPI, 12x12 figsize)
- **Processing Log Matrix**:
  - Real-time table display dengan localStorage persistence
  - Shows method, SNR, all 4 metrics
  - Excel export (XLSX) dari browser dengan SheetJS
  - Data persists across sessions

### 📓 Jupyter Notebook (`dtln_multi-method_evaluation.ipynb`)

- **Single Run Mode**: Proses satu file dengan preview lengkap
- **Batch Processing**: Multiple combinations (method × noise × SNR)
- **Excel Export (3 Sheets)**:
  - **All_Results**: Complete data 9 kolom (Method, Noise_Type, SNR_Input_dB, STOI, PESQ, MSE, MRE, Processing_Time_s, Realtime_Factor)
  - **Summary_By_Method**: Stats per metode (Mean ± Std)
  - **Summary_By_Noise**: Stats per noise type (Mean ± Std)
- **Google Colab Support**: Direct file download
- **Identical Configuration**: 100% sinkron dengan app.py (BLOCK_LEN=512, BLOCK_SHIFT=128, SAMPLE_RATE=16000)

---

## 🚀 Quick Start

### 1️⃣ Clone/Download Project

```powershell
cd "d:\Python\Project\DTLN Multi-Method Evaluation"
```

### 2️⃣ Install Dependencies (Python 3.12)

```powershell
pip install -r requirements.txt
```

**Dependencies** (latest versions for Python 3.12):

- Flask 3.1.0
- numpy 2.1.3
- scipy 1.14.1
- soundfile 0.12.1
- matplotlib 3.9.2
- onnxruntime 1.20.1
- pystoi 0.4.1
- pesq 0.0.4
- openpyxl 3.1.5

### 3️⃣ Setup ONNX Models

**Download DTLN models** dari: [DTLN Repository](https://github.com/breizhn/DTLN)

Letakkan di folder `pretrained_model/`:

```
DTLN Multi-Method Evaluation/
├── pretrained_model/
│   ├── model_1.onnx  ← Required
│   └── model_2.onnx  ← Required
```

### 4️⃣ Run Application

**Flask Web App**:

```powershell
python app.py
```

Buka: http://localhost:5000

**Jupyter Notebook**:

```powershell
jupyter notebook dtln_multi-method_evaluation.ipynb
```

Atau upload ke Google Colab

---

## 📖 Cara Penggunaan

### 🌐 Web App Mode

1. **Upload Clean Audio**

   - Format: WAV, FLAC, MP3
   - Akan di-resample ke 16kHz otomatis

2. **Pilih Metode**

   - **Deterministik**: Upload noise file, set SNR (e.g., 10 dB)
   - **Stokastik**: Pilih noise type (Gaussian/White/Random SNR/Mixed), set SNR
   - **Tradisional**: Pilih metode (Spectral Subtraction/Wiener Filter), upload noise, set SNR

3. **Process**

   - Klik "Process Audio"
   - Audio players muncul untuk mixed & processed
   - Spectrogram 2x2 ditampilkan
   - Metrics cards menampilkan 4 nilai (4 decimal places)
   - Log matrix terupdate otomatis

4. **Export**
   - Klik "Export Log to Excel" untuk download XLSX
   - Data tersimpan di localStorage (persistent)

### 📓 Notebook Mode

1. **Install & Upload Models** (Cell 1-3)

   ```python
   # Install packages
   !pip install flask numpy scipy soundfile matplotlib onnxruntime pystoi pesq openpyxl

   # Upload model_1.onnx dan model_2.onnx ke pretrained_model/
   ```

2. **Upload Audio Files** (Cell 4)

   ```python
   # Upload clean.wav dan noise.wav
   ```

3. **Single Run** (Cell 6)

   ```python
   method = "deterministik"  # atau "stokastik" atau "tradisional"
   snr_db = 10
   noise_type = "gaussian"  # untuk stokastik
   # Run cell untuk proses & visualisasi
   ```

4. **Batch Processing** (Cell 8)

   ```python
   # Define test matrix
   methods = ['deterministik', 'stokastik', 'tradisional']
   noise_types = ['gaussian', 'white']
   snr_values = [0, 5, 10, 15, 20]

   # Run cell untuk batch process & Excel export (3 sheets)
   ```

5. **Download Results** (Cell 10)
   ```python
   # Google Colab: Auto download
   # Local: Check results/ folder
   ```

---

## 📊 Interpretasi Metrik

### STOI (0.0000 - 1.0000)

| Score         | Interpretasi | Kualitas              |
| ------------- | ------------ | --------------------- |
| >0.8000       | Excellent    | Speech sangat jelas   |
| 0.6000-0.8000 | Good         | Speech cukup jelas    |
| <0.6000       | Poor         | Speech sulit dipahami |

### PESQ (-0.5000 - 4.5000)

| Score         | Interpretasi | Kualitas                       |
| ------------- | ------------ | ------------------------------ |
| >3.5000       | Excellent    | Perceptual quality sangat baik |
| 2.5000-3.5000 | Good         | Perceptual quality cukup baik  |
| <2.5000       | Poor         | Perceptual quality rendah      |

### MSE (Mean Squared Error)

- **<0.0100**: Excellent
- **0.0100-0.0500**: Good
- **>0.0500**: Poor
- **Lower is better** (perbedaan sinyal lebih kecil)

### MRE (Mean Relative Error)

- **<0.1000**: Excellent
- **0.1000-0.3000**: Good
- **>0.3000**: Poor
- **Lower is better** (error relatif lebih kecil)

---

## 🏗️ Struktur Project

```
DTLN Multi-Method Evaluation/
├── app.py                          # Flask backend (main)
├── dtln_multi-method_evaluation.ipynb  # Jupyter notebook (batch)
├── requirements.txt                # Python deps (3.12 compatible)
├── README.md                       # Dokumentasi lengkap
├── setup.py                        # Setup script
├── .gitignore                      # Git config (ignores pretrained_model/)
│
├── pretrained_model/               # ONNX models folder
│   ├── model_1.onnx               # DTLN stateful model
│   └── model_2.onnx               # DTLN stateless model
│
├── templates/
│   └── index.html                 # Tailwind UI (shadcn theme)
│
├── uploads/                        # Uploaded files (auto-created)
├── outputs/                        # Processed audio (auto-created)
└── static/
    └── spectrograms/              # Combined 2x2 PNG (auto-created)
└── results/
    ├── audio/                     # Batch processed audio
    ├── spectrograms/              # Batch spectrograms
    ├── metrics/                   # Individual metrics
    └── batch_results.xlsx         # Excel (3 sheets)
```

---

## 🔧 Konfigurasi Teknis

### Audio Parameters (matches DTLN requirements)

```python
SAMPLE_RATE = 16000    # Fixed untuk DTLN (16kHz)
BLOCK_LEN = 512        # FFT window size
BLOCK_SHIFT = 128      # Hop size (25% overlap)
```

### Web App Limits

```python
MAX_FILE_SIZE = 50 MB
ALLOWED_FORMATS = ['wav', 'flac', 'mp3']
```

### Processing Methods

#### 1. Deterministik (Fixed Noise + DTLN)

```python
# Noise tetap dari uploaded file
mixed = add_gaussian_noise(clean, noise, snr_db)
processed = process_dtln(mixed, model_1, model_2)
```

**Use Case**: Reproducible baseline, requires uploaded noise file

#### 2. Stokastik (Random Noise + DTLN)

```python
# Noise random setiap run
noise_types = ['gaussian', 'white', 'random_snr', 'mixed']
mixed = add_white_noise(clean, snr_db)  # contoh
processed = process_dtln(mixed, model_1, model_2)
```

**Use Case**: Robustness testing, synthetic noise generation

#### 3. Tradisional (DSP Methods)

```python
# No neural network
methods = ['spectral_subtraction', 'wiener_filter']
processed = spectral_subtraction(mixed, noise_profile)
# atau
processed = wiener_filter(mixed, noise_profile)
```

**Use Case**: Baseline comparison, faster inference

### Spectrogram Configuration

```python
# Combined 2x2 layout
fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=100)
plt.pcolormesh(t, f, Sxx, shading='gouraud', cmap='magma')
```

**Colormap**: `magma` (gamma style, NOT viridis/green)

---

## 🎓 Referensi & Credits

### Paper & Code

- **DTLN**: [Westhausen & Meyer, "Dual-Signal Transformation LSTM Network"](https://github.com/breizhn/DTLN)
- **ONNX Runtime**: [Microsoft ONNX Runtime](https://onnxruntime.ai/)

### Libraries

- **Flask 3.1.0**: Web framework
- **ONNX Runtime 1.20.1**: Model inference
- **pystoi 0.4.1**: STOI metric calculation
- **pesq 0.0.4**: PESQ metric calculation
- **matplotlib 3.9.2**: Spectrogram visualization
- **openpyxl 3.1.5**: Excel export

### UI Framework

- **Tailwind CSS 3.4**: Utility-first CSS
- **shadcn/ui**: Design system (slate/blue/purple/amber/rose palette)
- **SheetJS (xlsx 0.18.5)**: Browser-side Excel export

---

## 🐛 Troubleshooting

### ❌ "ONNX models not found"

**Solusi**: Letakkan `model_1.onnx` dan `model_2.onnx` di folder `pretrained_model/`

### ❌ "This model only supports 16000Hz sampling rate"

**Solusi**: Audio di-resample otomatis, pastikan scipy terinstall:

```powershell
pip install scipy==1.14.1
```

### ❌ PESQ Error

**Solusi**: PESQ sensitif terhadap format, gunakan audio 16kHz:

```python
from scipy import signal
audio_16k = signal.resample(audio, int(len(audio) * 16000 / sr))
```

### ❌ Import Error (numpy, onnx, etc.)

**Solusi**: Install ulang dengan Python 3.12:

```powershell
pip install -r requirements.txt --upgrade
```

### ❌ Spectrogram tidak muncul

**Solusi**: Cek folder `static/spectrograms/` dibuat otomatis:

```python
os.makedirs('static/spectrograms', exist_ok=True)
```

### ❌ Excel export gagal (web app)

**Solusi**: Cek console browser, pastikan SheetJS CDN loaded:

```html
<script src="https://cdn.sheetjs.com/xlsx-0.18.5/package/dist/xlsx.full.min.js"></script>
```

---

## 🚀 Pengembangan Lanjutan

### Custom Noise Dataset

1. Buat folder `noise_dataset/`
2. Tambahkan file `.wav` dengan naming convention:
   ```
   noise_dataset/
   ├── gaussian_01.wav
   ├── white_01.wav
   ├── traffic_01.wav
   └── ...
   ```
3. Modifikasi `app.py`:
   ```python
   def add_custom_noise(clean, noise_type, snr_db):
       noise_file = f'noise_dataset/{noise_type}_01.wav'
       noise, _ = sf.read(noise_file)
       # Mix dengan SNR
   ```

### New Metrics

Tambahkan di `calculate_metrics()`:

```python
from scipy.stats import pearsonr

def calculate_metrics(clean, processed):
    # Existing metrics...

    # Add Pearson correlation
    corr, _ = pearsonr(clean, processed)
    metrics['correlation'] = round(corr, 4)

    return metrics
```

### Custom DTLN Models

Ganti model di `pretrained_model/`:

```python
# Train model baru dengan DTLN repo
# Export ke ONNX
# Replace model_1.onnx dan model_2.onnx
```

---

## 📄 Lisensi

MIT License (sesuai dengan DTLN reference code)

## 👨‍💻 Author

Generated for **DTLN Multi-Method Audio Evaluation Project**

## 🙏 Acknowledgments

- **Nils L. Westhausen** untuk DTLN model dan reference implementation
- **Flask** framework team
- **ONNX Runtime** team
- **Tailwind CSS** & **shadcn/ui** design system
- Semua open-source libraries yang digunakan

---

**Status**: ✅ Production Ready  
**Last Updated**: 2024  
**Python**: 3.12+  
**License**: MIT
