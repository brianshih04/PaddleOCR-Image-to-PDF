import logging
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

@dataclass
class NormalizedPage:
    index: int
    image_rgb: np.ndarray
    original_size: tuple[int, int]  # (width, height)
    is_pdf: bool

def parse_magic_number(filepath: Path) -> str:
    with open(filepath, "rb") as f:
        magic = f.read(4)
    if magic.startswith(b"%PDF"):
        return "pdf"
    elif magic.startswith(b"\xFF\xD8"):
        return "jpeg"
    elif magic.startswith(b"\x89PNG"):
        return "png"
    elif magic.startswith(b"BM"):
        return "bmp"
    elif magic.startswith(b"II*\x00") or magic.startswith(b"MM\x00*"):
        return "tiff"
    return "unknown"

def load_data(filepath: str | Path, dpi: int = 300) -> list[NormalizedPage]:
    """
    Stage 1: Input Routing and Normalization.
    Takes a path and routes it to PyMuPDF or OpenCV based on Magic Number.
    Returns a sequence of standard (Width, Height, RGB_Array) memory matrices.
    """
    filepath = Path(filepath)
    file_type = parse_magic_number(filepath)
    
    pages = []
    
    if file_type == "pdf":
        doc = fitz.open(filepath)
        for i in range(len(doc)):
            page = doc[i]
            # Convert PDF Page to RGB Pixmap
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
            
            # Buffer to contiguous NumPy structured array
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
            # PyMuPDF buffer is read-only, make a copy to strictly own the memory
            img_array = np.ascontiguousarray(img_array.copy())
            
            pages.append(NormalizedPage(index=i, image_rgb=img_array, original_size=(pix.width, pix.height), is_pdf=True))
            
        doc.close()
    else:
        # Fallback to OpenCV (handles JPEG, PNG, TIFF, BMP)
        # Note: Multi-page TIFF decoding via cv2.imreadmulti
        ret, images = cv2.imreadmulti(str(filepath), flags=cv2.IMREAD_COLOR)
        if not ret or not images:
            # Fallback single image
            img = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
            if img is not None:
                images = [img]
            else:
                raise ValueError(f"Failed to decode image: {filepath}")
                
        for i, img in enumerate(images):
            # OpenCV loads as BGR; pipeline strictly requires RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_img = np.ascontiguousarray(rgb_img)
            h, w = rgb_img.shape[:2]
            
            pages.append(NormalizedPage(index=i, image_rgb=rgb_img, original_size=(w, h), is_pdf=False))

    return pages
