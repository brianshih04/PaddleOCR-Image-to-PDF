import logging
import time
from pathlib import Path
from dataclasses import dataclass
import gc

from rasterizer import load_data, NormalizedPage
from ocr_engine import PaddleOcrEngine
from coord_mapper import CoordMapper
from pdf_writer import PDFReconstructor

logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    elapsed_seconds: float
    total_pages: int
    output_path: Path

def run_pipeline(
    input_path: Path, 
    output_path: Path, 
    det_model: Path, 
    rec_model: Path, 
    dict_path: Path, 
    dpi: int = 300
) -> PipelineResult:
    t_start = time.monotonic()
    
    # Init OCR Engine (ONNX Runtime wrapper)
    logger.info("Initializing ONNX Text Engine...")
    ocr_engine = PaddleOcrEngine(
        det_model_dir=str(det_model),
        rec_model_dir=str(rec_model),
        dict_path=str(dict_path)
    )
    
    # 1. Image loading / PDF Rasterizing
    logger.info(f"Decoding {input_path} into normalized contiguous matrices...")
    pages: list[NormalizedPage] = load_data(input_path, dpi=dpi)
    total_pages = len(pages)
    
    pdf_reconstructor = PDFReconstructor()
    
    for i, page in enumerate(pages):
        logger.info(f"--- Processing Page {i+1}/{total_pages} ---")
        
        # 2a. Text Detection
        polygons = ocr_engine.detect_text_polygons(page.image_rgb)
        logger.info(f"Detected {len(polygons)} text regions.")
        
        extracted_blocks = []
        for poly in polygons:
            # 3. Coordinate mapping
            x_min, y_min, x_max, y_max = CoordMapper.polygon_to_orthogonal_bbox(poly)
            # Clip bbox to image bounds
            h, w = page.image_rgb.shape[:2]
            y0, y1 = max(0, int(y_min)), min(h, int(y_max))
            x0, x1 = max(0, int(x_min)), min(w, int(x_max))
            
            # 2b. Cropping and Recognition
            # Placeholder strict geometric cropping instead of actual perspective transform
            cropped = page.image_rgb[y0:y1, x0:x1]
            if min(cropped.shape[:2]) > 0:
                text, conf = ocr_engine.recognize_text(cropped)
                
                # Append to block list for embedding
                # 1:1 Scale
                if text.strip() != "":
                    extracted_blocks.append((text, (x_min, y_min, x_max, y_max)))
        
        # 4. Synthesize final PDF structure via PyMuPDF 
        pdf_reconstructor.add_page(page.image_rgb, extracted_blocks)
        
        # Explicit garbage collection to tightly control RAM peak usage
        del page
        gc.collect()

    pdf_reconstructor.save(output_path)
    
    t_end = time.monotonic()
    return PipelineResult(
        elapsed_seconds=float(t_end - t_start),
        total_pages=total_pages,
        output_path=output_path
    )
