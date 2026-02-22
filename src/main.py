import argparse
import logging
import sys
from pathlib import Path

from pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="PaddleOCR ONNX Runtime Fast Searchable PDF Engine")
    
    parser.add_argument("input", type=str, help="Input PDF or image filepath")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output PDF filepath")
    parser.add_argument("--det", type=str, default="../models/det.onnx", help="Path to det.onnx model")
    parser.add_argument("--rec", type=str, default="../models/rec.onnx", help="Path to rec.onnx model")
    parser.add_argument("--dict", type=str, default="../models/ppocr_keys_v1.txt", help="Path to dictionary file")
    parser.add_argument("--dpi", type=int, default=300, help="PDF rasterization DPI")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose Logging")
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("main")
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
        
    output_path = Path(args.output) if args.output else input_path.parent / f"{input_path.stem}_searchable.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    det_model = Path(__file__).parent / args.det
    rec_model = Path(__file__).parent / args.rec
    dict_path = Path(__file__).parent / args.dict
    
    try:
        res = run_pipeline(
            input_path=input_path,
            output_path=output_path,
            det_model=det_model,
            rec_model=rec_model,
            dict_path=dict_path,
            dpi=args.dpi
        )
        logger.info(f"Done! Evaluated {res.total_pages} pages in {res.elapsed_seconds:.2f} seconds.")
        logger.info(f"Output saved to: {res.output_path}")
        return 0
    except Exception as e:
        logger.exception("Inference failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
