# PaddleOCR Image-to-Searchable PDF Engine (ONNX) 🚀

A high-performance, offline, multi-language desktop application that converts scanned documents and images (PDF, JPEG, PNG, TIFF, BMP) into precisely aligned **Searchable PDFs**. Built with PySide6, ONNX Runtime, and PaddleOCR v4.

## ✨ Key Features

* **⚡ Ultra-Fast ONNX Offline Engine**: Uses `ONNX Runtime` instead of the bulky Python PaddlePaddle framework, ensuring lightweight CPU/GPU inference, low memory usage, and easy binary packaging.
* **🔍 Pixel-Perfect Searchable PDFs**: Integrates `PyMuPDF` with advanced matrix morphing to stretch and perfectly align invisible OCR text directly over the original image pixels, ensuring exact Adobe Acrobat search highlights.
* **🧠 NLP Auto-Classification**: Includes a built-in HuggingFace Transformer (`MiniLM-L6-v2`) and a Regex Rule Engine to automatically categorize processed documents into subfolders based on text semantics.
* **📁 Hot Folder Background Daemon**: Monitors a selected directory in the background (`watchdog`). Automatically queues, processes, and intelligently files incoming scans without any manual clicks.
* **🌍 Multi-Language Hot-Swapping**: Automatically manages VRAM. Supports dropping in official PaddleOCR `rec` models (e.g., Traditional Chinese, English, Japanese, Korean) for instant multi-language support.
* **🎨 Modern GUI**: A responsive PyQt/PySide6 multi-threaded interface preventing UI freezes during heavy AI workloads.

## 📦 File Output Strategy

* **High Compression**: Output PDFs are maximally compressed (`garbage=4`, `deflate=True`), retaining visual quality while minimizing storage.
* **Metadata Injection**: Injects the automatically classified NLP tags and OCR language signatures deeply into the PDF's internal metadata (`/Keywords`, `/Subject`).
* **Atomic Safeties**: Original raw image files are moved out of the Hot Folder to `_Backup` or `_Error` folders to prevent infinite processing loops.

## 🛠️ Installation & Setup

### 1. Requirements

Ensure you have Python 3.10+ installed.

```bash
pip install -r requirements.txt
```

### 2. Download Official OCR AI Models

We've provided a bootstrap script that pulls the required text detection and recognition weights (converted to `.onnx`) directly from HuggingFace to bypass local PaddlePaddle installation complexities.

```bash
python src/setup_models.py
```

*This will download `ch_PP-OCRv4` det and rec models, plus dictionaries into the `models/` folder.*

### 3. (Optional) NLP Auto-Classification Models

If you wish to use the neural network classification fallback:

1. Download [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main) ONNX weights.
2. Place `model.onnx` and `tokenizer.json` inside `models/nlp/all-MiniLM-L6-v2/`.

## 🚀 Usage

Start the graphical interface:

```bash
python src/gui_main.py
```

1. **Input Mode**: Either Drag & Drop files into the queue, OR select a "Hot Folder" directory to monitor continuously.
2. **Output Path**: Choose where the Searchable PDFs should be saved.
3. **Classification**: Toggle "Enable Smart Classification" and define your own Custom Rules.
4. **Action**: Click "Start Processing" to let the dual-thread QWorker burn through the queue.

## ⚙️ Architecture

* **`ocr_engine.py`**: DBNet (Vatti clipping/dilation) + SVTR_LCNet (CTC sequential decoding) purely via ONNX Runtime.
* **`coord_mapper.py`**: Spatial transforms mapping rasterized pixels to Cartesian Points (1:1 geometric scale).
* **`pdf_writer.py`**: `PyMuPDF` synthesizer layering invisible CJK vector fonts using `fitz.Matrix` horizontal scaling.
* **`classifier.py`**: Transformers Tokenizer + Regex string matching router.
* **`monitor.py`**: `watchdog` daemon with smart file-lock probing (prevents reading half-written scans).

## 📄 License

This wrapper application is open-source. The underlying PaddleOCR network architectures are licensed under Apache 2.0 by PaddlePaddle.
