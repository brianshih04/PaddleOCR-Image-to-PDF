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
            text_rect = fitz.Rect(x0, y0, x1, y1)
            
            # Using Shape object for explicit text insertion styling
            # fitz.insert_textbox automatically handles layout inside the rect, 
            # we specify render_mode=3 via textwriter for precise control
            
            tw = fitz.TextWriter(page.rect)
            # Create a transparent text
            # render_mode: 0=Fill, 1=Stroke, 2=FillThenStroke, 3=Invisible
            font = fitz.Font("cjk") # Fallback to builtin CJK
            tw.append(
                text_rect.bl, # Bottom left text origin (approximate)
                text,
                font=font,
                fontsize=max(int(y1 - y0), 1) * 0.8 # Dynamic sizing
            )
            tw.write_text(page, render_mode=3) # Critical parameter: invisible text for selection

    def save(self, filepath: str | Path):
        """
        Saves the PDF with strict garbage collection and stream deflation
        to keep the searchable PDF footprint minimal.
        """
        self.doc.save(str(filepath), garbage=4, deflate=True)
        self.doc.close()
        logger.info(f"Reconstructed PDF saved securely at {filepath}")
