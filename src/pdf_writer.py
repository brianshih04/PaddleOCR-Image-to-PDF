import logging
import fitz # PyMuPDF
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class PDFReconstructor:
    """
    Stage 4: Dynamic PDF Recreation.
    Using PyMuPDF to create blank PDF, insert base image at Z_Index 0,
    and insert render_mode=3 invisible vector text at Z_Index 1.
    """
    def __init__(self):
        self.doc = fitz.open()
        
    def set_metadata(self, label: str, language: str):
        """
        Inject classification label and OCR language into the PDF metadata 
        for OS-level or search engine indexing.
        """
        meta = self.doc.metadata
        if not meta:
            meta = {}
            
        meta["keywords"] = f"OCR, {label}, {language}"
        meta["subject"] = label
        meta["title"] = f"Auto-Routed {label} Document"
        
        self.doc.set_metadata(meta)
        logger.info(f"Injected PDF Metadata: Subject={label}, Language={language}")
        
    def add_page(self, img_array: np.ndarray, text_blocks: list[tuple[str, tuple[float, float, float, float]]]):
        """
        text_blocks: list of (Text_String, (x_min, y_min, x_max, y_max)) in pixels
        """
        # H, W, C
        h, w, _ = img_array.shape
        
        # 1. Create a page strictly matching the pixel dimensions (1 pixel = 1 pt rendering)
        page = self.doc.new_page(width=w, height=h)
        rect = fitz.Rect(0, 0, w, h)
        
        # PyMuPDF expects RGB samples in continuous bytes
        samples = np.ascontiguousarray(img_array).tobytes()
        pix = fitz.Pixmap(fitz.csRGB, w, h, samples, False)
        
        # Z-index 0: Insert Image 
        page.insert_image(rect, pixmap=pix)
        
        # Z-index 1: Insert Invisible Text
        # PyMuPDF doesn't explicitly have z-index, but operations run in order of draw.
        # render_mode=3 guarantees visibility is zero but selectable.
        for text, (x0, y0, x1, y1) in text_blocks:
            box_w = max(x1 - x0, 1)
            box_h = max(y1 - y0, 1)
            
            tw = fitz.TextWriter(page.rect)
            font = fitz.Font("cjk") # builtin CJK
            fontsize = box_h * 0.8
            
            # Calculate geometric ratio to horizontally stretch text safely
            tl = font.text_length(text, fontsize=fontsize)
            tl = max(tl, 1)
            ratio = float(box_w / tl)
            
            # Origin point mapped to approx text baseline
            base_y = y1 - box_h * 0.15
            pt = fitz.Point(x0, base_y)
            
            tw.append(pt, text, font=font, fontsize=fontsize)
            
            # Stretch horizontally using a Matrix
            mat = fitz.Matrix(ratio, 0, 0, 1, 0, 0)
            tw.write_text(page, render_mode=3, morph=(pt, mat)) # render_mode=3: invisible text

    def save(self, filepath: str | Path):
        """
        Saves the PDF with strict garbage collection and stream deflation
        to keep the searchable PDF footprint minimal.
        """
        self.doc.save(str(filepath), garbage=4, deflate=True)
        self.doc.close()
        logger.info(f"Reconstructed PDF saved securely at {filepath}")
