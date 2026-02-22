# PaddleOCR 異質影像轉可搜尋 PDF 系統 (Intel OpenVINO 架構) 🚀

這是一套**高效能、全離線、支援多國語言**的 Windows 桌面應用程式。旨在將各式掃描文件與影像（包含 PDF, JPEG, PNG, TIFF, BMP）精準轉換為**可搜尋、可選取文字的 PDF (Searchable PDF)**。
本系統底層使用 PySide6 打造非同步圖形介面，並以 **Intel OpenVINO** 作為核心推論引擎，搭載經過最佳化的 PaddleOCR 模型，實現伺服器等級的離線推論速度。

## ✨ 核心特色

* **⚡ 超高速 OpenVINO 離線引擎**: 徹底拔除臃腫的 Python 機器學習框架 (如 PyTorch / PaddlePaddle)。直接使用 Intel 原生開發的 `openvino` 引擎，強制鎖定 **CPU 多核心並行解碼**（在動態張量長度裁切下，實測大勝 GPU 編譯效能，每頁僅需 ~3.5 秒），保證無損極速量產，且不挑顯示卡。
* **🔍 像素級精準的 Searchable PDF**: 整合 `PyMuPDF` 結合矩陣變換演算法，將辨識出的隱藏向量文字，精準映射並縮放拉伸至原始影像的像素座標上，確保游標選取或 Adobe Acrobat 搜尋時的絕對貼合對齊。
* **🧠 NLP 雙軌自動分類引擎**: 內建自訂的「正規表示式 (Regex)」規則引擎，並搭載輕量級的 HuggingFace NLP 語意分析模型 (`MiniLM-L6-v2`)。掃描完成後自動判讀內文，將產出的 PDF 自動路由分類至對應的子資料夾。
* **📁 智慧熱區監控防護 (`watchdog`)**: 背景守護進程可 24 小時監控指定的「熱區 (Hot Folder)」。自動攔截來自掃描器的新檔案，並具備**高階檔案寫入鎖定探測 (File-lock Probing)**，絕對不讀取寫入到一半的破壞檔案。
* **🌍 多國語言熱插拔切換**: 支援多種官方 PaddleOCR 辨識模型（如：繁體中文、簡體中文、日文、韓文、英文等）。UI 介面一鍵切換，背景自動進行 VRAM / 記憶體資源的銷毀與釋放。
* **🎨 現代化非同步 GUI**: 基於 PySide6 開發的多執行緒介面，將繁重的 AI 運算陣列與磁碟 I/O 完全封裝在背景 QThread 中，確保主視窗的操作永遠如絲般滑順，絕不凍結。

## 📦 檔案輸出與歸檔防錯

* **極致壓縮技術**: 輸出的 PDF 強制啟用最高壓縮層級 (`garbage=4`, `deflate=True`)，在保留最高畫質的同時，將占用空間縮減到極致。
* **詮釋資料 (Metadata) 注入**: 將 NLP 預測出的文件類別與 OCR 語言屬性，無縫寫入 PDF 內部的 Metadata 標籤內 (`/Keywords`, `/Subject`)，利於企業知識庫建立。
* **原子搬移死結迴避**: 處理完成後，原始圖檔將被自動備份至 `_Backup` 資料夾，產出的 PDF 則送至歸檔區；若遭遇毀損檔，則一律隔離至 `_Error` 資料夾，徹底確保監控熱區淨空，防止巡圈死結發生。

## 🛠️ 安裝與啟動方式

### 方案 A：直接下載免安裝版 (Standalone EXE)

前往 [Releases] 區下載最新編譯打包好的 `paddleocr-engine.exe`。
此版本已將本機 Python 環境、Intel OpenVINO 動態連結庫、PySide6 介面與所有的中英文 OCR 模型全數封裝為單一執行檔。放置於任何 Windows 電腦，**雙擊即可立刻運作，不需安裝任何軟體**。

### 方案 B：開發者源碼執行環境部署

若您希望修改程式碼：
請確保系統已具備 Python 3.10+ 環境：

```bash
pip install -r requirements.txt
```

#### 部署所需模型

本專案執行需依賴轉換後的 `.onnx` 模型：

1. **OCR 模型 (`models/`)**：需包含 `det.onnx` (通用文字偵測) 以及 `rec/` 目錄下的各國語言辨識模型。
2. **NLP 分類模型 (`models/nlp/`)**：需包含提取特徵長度的 `model.onnx` 及截斷分詞器 `tokenizer.json`。
3. **字典文件 (`models/dict/`)**：提供字集映射解碼的 `.txt` 檔案。

直接執行主程式：

```bash
python src/gui_main.py
```

## 🚀 操作流程

1. **輸入模式選擇**: 直接將檔案或是資料夾「拖放」進工作佇列；或是在右側面板勾選「啟用背景熱區監控 (Hot Folder)」。
2. **設定輸出目的地**: 選擇 Searchable PDF 儲存的絕對路徑。
3. **語意分類 (選項)**: 打開「啟用智能分類」並配置自訂的比對規則 (`rules.json`)。
4. **一鍵啟動**: 點選「開始轉換」，讓背景的 QWorker 執行緒全速吞吐您的排隊檔案。

## ⚙️ 系統核心架構

* **`hw_detect.py`**: 利用 `ov.Core()` 進行底層探測。由於 Intel GPU 內顯在面對文字偵測不定長的「動態張量」時會頻繁觸發痛苦的 OpenCL 重編譯瓶頸，本引擎已**永久鎖定為 CPU 實體核心並行提供者**，榨出極速定頻效能。
* **`ocr_engine.py`**: DBNet (四點座標預測) + SVTR_LCNet (CTC 時序解碼)，全數依託於 OpenVINO 運作。
* **`coord_mapper.py`**: 掌管 Raster Pixel (點陣) 至 Cartesian Points (笛卡爾空間) 的 1:1 幾何映射轉換。
* **`pdf_writer.py`**: 使用 `fitz.Matrix` 將萃取出的文字作為浮動底層的向量字體，無痕疊加於原始圖紙之上。

## 📄 授權條款

本桌面工具專案實作開源。底層依託的深度學習網路權重 (PaddleOCR v4) 遵循 PaddlePaddle 官方的 Apache License 2.0 條款。
