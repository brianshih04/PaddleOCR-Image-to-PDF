import os
import urllib.request
from pathlib import Path

# Pre-converted ONNX Models from RapidOCR (HuggingFace)
MODELS = {
    "models/det.onnx": "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_det_infer.onnx",
    "models/rec/ch_PP-OCRv4_rec.onnx": "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx"
}

DICTS = {
    "models/dict/ppocr_keys_v1.txt": "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/ppocr_keys_v1.txt"
}

def download_file(url, out_path):
    print(f"Downloading {out_path.name} ...")
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, context=ctx) as response, open(out_path, 'wb') as out_file:
        data = response.read() # a `bytes` object
        out_file.write(data)

def main():
    base_dir = Path(__file__).parent.parent
    
    # Create directories
    (base_dir / "models").mkdir(exist_ok=True)
    (base_dir / "models" / "rec").mkdir(exist_ok=True)
    (base_dir / "models" / "dict").mkdir(exist_ok=True)
    
    print("Downloading pre-converted ONNX models...")
    for out_rel_path, url in MODELS.items():
        out_file = base_dir / out_rel_path
        if not out_file.exists():
            download_file(url, out_file)
            print(f"✅ Saved to {out_file}")
        else:
            print(f"⏭️  Already exists: {out_file}")
            
    print("\nDownloading dictionaries...")
    for out_rel_path, url in DICTS.items():
        out_file = base_dir / out_rel_path
        if not out_file.exists():
            download_file(url, out_file)
            print(f"✅ Saved to {out_file}")
        else:
            print(f"⏭️  Already exists: {out_file}")
            
    print("\n🎉 OCR Engine models are fully installed and ready!")

if __name__ == "__main__":
    main()
